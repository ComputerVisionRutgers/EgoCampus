import os, sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import io
import math
import re
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import line_profiler
import matplotlib.pyplot as plt
import numpy as np
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import polars as pl
import torch
import torchvision
import ubelt as ub
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from PIL import Image
from jaxtyping import Float
from natsort import natsorted
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import GaussianBlur
from tqdm import tqdm

from . import utils
from .utils import parse_args, load_config, load_toml_config


class Direction(Enum):
    'probably unimportant but i wanted to save memory by using an int instead of a string' #TODO remove this
    FORWARD=1
    REVERSE=2

    @classmethod
    def from_str(cls, str):
        str = str.lower()
        if str == 'forward' or str == 'fwd':
            return cls.FORWARD
        elif str == 'reverse' or str == 'rev':
            return cls.REVERSE
        else:
            raise ValueError(f'{str} is invalid value for Direction enum')

    def __repr__(self):
        if self is Direction.FORWARD:
            return "forward"
        else:
            return "reverse"

    def to_str(self):
        warnings.warn("dont use Direction.to_str(), use repr")
        return self.__repr__()

@dataclass
class ClipDesc:
    path: int
    subject: str
    direction: Direction | str
    frames: list[int]

    def __repr__(self):
        return f'subject {self.subject}, path {self.path} {str(self.direction)}, frames {self.frames}'

class GaussianGenerator():
    def __init__(
        self,
        cfg: dict,
        mask: Optional[Image.Image] = None, # TODO mask all gaussians if supplied
        device: torch.device | str = 'cpu',
    ):
        self.imsize = cfg['data']['output_size']
        self.W, self.H = self.imsize
        sigma = cfg['data']['gaussian_sigma']
        
        if self.W != self.H:
            print("the gaussian scales with width, but to handle this it would have to scale with the diagonal of the image, or some other approach (min dim?)")
            raise NotImplementedError

        normalization_method = cfg['data']['normalization_method'].lower()
        match normalization_method:
            case 'l1':
                self.normalization = lambda x: x / x.sum()
            case 'max-scaling':
                self.normalization = lambda x: x / x.max()
            case 'none':
                self.normalization = lambda x: x
            case _:
                raise NotImplementedError

        if str(cfg['data']['use_mask']).lower() == 'true':
            assert mask is not None
            pil_to_tensor = transforms.PILToTensor()
            self.mask = (pil_to_tensor(mask) > 0).squeeze(0).to(device)
        else:
            self.mask = None

        gaussian_img = torch.zeros(2*self.W-1, 2*self.H-1).to(device)
        gaussian_img[self.W-1,self.H-1] = 1.0
        gaussian_img = gaussian_img.unsqueeze(0)

        kernel_w, kernel_h = self.W, self.H
        if kernel_w % 2 == 0:
            kernel_w += 1
        if kernel_h % 2 == 0:
            kernel_h += 1

        tf = GaussianBlur((kernel_w, kernel_h), sigma*self.W/1000)
        self.gaussian_img = tf(gaussian_img)
        
    def __call__(self, center: torch.Tensor):
        # TODO make this cleaner
        start_x = int(round(((1-center[0]) * self.W).item()))
        start_y = int(round(((1-center[1]) * self.H).item()))

        ret = self.gaussian_img[0, start_y:start_y+self.H, start_x:start_x+self.W]
        assert ret.min() >= 0

        if self.mask is not None:
            ret *= self.mask

        return self.normalization(ret)

class EgoCampusDataset(Dataset):
    def __init__(self, cfg, decoding_device: torch.device | str = 'cpu'):
        super().__init__()
        mode = cfg['mode']
        print(f"Loading EgoCampus {mode} split!")

        self.mode = mode
        self.cfg = cfg
        self.decoding_device = decoding_device

        self.context_window = cfg['data']['context_window']
        self.sample_stride = cfg['data']['sample_stride']
        self.window_stride = cfg['data']['window_stride']
        self.cache_image_bytes = str(cfg['data']['cache_image_bytes']).lower() == 'true'

        if self.cache_image_bytes:
            print("caching image bytes")

        excluded_clips = set()
        if 'excluded_clips_path' in cfg['data']:
            with open(cfg['data']['excluded_clips_path'], 'r') as f:
                csv_reader = csv.DictReader(f)
                for row in csv_reader:
                    excluded_clips.add((int(row['path']), row['subject'], row['direction']))

            print(f'excluding {len(excluded_clips)} videos from the dataset')

        self.mean = torch.tensor(self.cfg['data']['mean'])
        self.std = torch.tensor(self.cfg['data']['std'])
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.unnormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist()) # utility

        self.df = pl.DataFrame(schema = {
            "path": pl.Int32,
            "subject": str,
            "direction": str,
            "images_dir": str,
            "image_names": list[str],
        })

        self.image_byte_cache = {}
        self.gaze_cache = {}

        path_prefix = ub.Path(cfg['data']['path_prefix'])
        dataset_walk = list(os.walk(str(path_prefix)))

        if len(dataset_walk) == 0:
            print(f'No dataset found at {path_prefix}')

        # get the bitmap mask
        _, _, top_level_files = dataset_walk[0]
        for file in top_level_files:
            if file.endswith('.bmp'):
                self.mask = Image.open(path_prefix / file).convert('L')
        
        if str(cfg['data']['use_mask']).lower() == 'true':
            assert self.mask is not None, 'no mask found'
        else:
            self.mask = None

        self.gaussian_generator = GaussianGenerator(
            self.cfg,
            device=self.decoding_device,
            mask=self.mask
        )

        # iterate over the rest of the dataset directory
        missing_gaze_info = []
        dir_pattern = re.compile(r'.*path(.*)\/subject(.*)\/(:?forward|reverse)')
        for i, (root, dirs, files) in tqdm(enumerate(dataset_walk[1:]), total=len(dataset_walk)-1, desc='indexing data'):
            if not dirs == ['images']:
                continue
            
            images_dir = os.path.join(root, 'images')
            image_names = dataset_walk[i+2][2] # 'files' from the next index

            dir_matches = dir_pattern.match(root.lower())
            if dir_matches is None:
                raise Exception(f'no regex matches for {root=}')
            path = int(dir_matches.group(1))
            subject = dir_matches.group(2).lstrip('0')
            direction = dir_matches.group(3)

            if not path in cfg['data'][mode]['included_paths']:
                continue

            if path in cfg['data']['excluded_paths'] or subject in cfg['data']['excluded_subjects']:
                continue

            if (path, subject, direction) in excluded_clips:
                continue

            if not 'gaze.npy' in files:
                missing_gaze_info.append(f's{subject}_p{path}_{direction[0]}')
                continue
            gaze_data = torch.from_numpy(np.load(os.path.join(root, 'gaze.npy')).astype(np.float32)) # adds 7 seconds, annoyingly

            if self.cache_image_bytes:
                def _image_worker(filepath):
                    with open(filepath, "rb") as f:
                        return f.read()

                image_paths = [os.path.join(root, f"images/{im_name}") for im_name in image_names][cfg['data']['exclude_first_n_frames']:]
                with ThreadPoolExecutor(max_workers=cfg['data']['num_workers']) as executor:
                    image_bytes = list(executor.map(_image_worker, image_paths))

                self.image_byte_cache[(path, subject, direction)] = image_bytes


            self.gaze_cache[(path, subject, direction)] = gaze_data

            self.df.vstack(pl.DataFrame({
                "path": path,
                "subject": subject,
                "direction": direction,
                "images_dir": images_dir,
                "image_names": [natsorted(image_names)[cfg['data']['exclude_first_n_frames']:]], # natsort adds 5 seconds
            }), in_place=True)

        self.df = self.df.rechunk()

        if len(missing_gaze_info) > 0:
            warnings.warn(f"Missing gaze info for the following videos {missing_gaze_info}")
        
        frames_per_video = np.array(self.df.select(
            frames_per_video=pl.col("image_names").list.len()
        ).rows())[:, 0]
        self.frames_per_video = frames_per_video

        self.items_per_video = np.array(
            [len(range(0, num_frames-self.context_window, self.window_stride)) for num_frames in frames_per_video]
        )
        self.items_per_video_acc = self.items_per_video.cumsum()

    def idx_to_clip_desc(self, idx):
        video_idx = np.argmax((self.items_per_video_acc - idx) > 0)
        if video_idx == 0:
            frame_idx = idx * self.window_stride
        else:
            frame_idx = (idx - self.items_per_video_acc[video_idx-1]) * self.window_stride

        return ClipDesc(
            *self.df.select("path", "subject", "direction").row(video_idx),
            list(range(frame_idx, frame_idx+self.context_window, self.sample_stride))
        )

    def get_item_from_desc(self, desc: ClipDesc):
        potential_rows = self.df.filter(path=desc.path, subject=desc.subject, direction=desc.direction)

        assert potential_rows.height <= 1, 'multiple dataset entries with identical path, subject, direction. this should not happen'
        assert potential_rows.height == 1, f'no item found with desc {desc}'

        return self.getitem_helper(desc, potential_rows.row(0, named=True))

    @line_profiler.profile
    def __getitem__(self, idx, for_eii=False):
        desc = self.idx_to_clip_desc(idx)
        row = self.df.row(np.argmax((self.items_per_video_acc - idx) > 0), named=True)

        return self.getitem_helper(desc, row, for_eii=for_eii)

    @line_profiler.profile
    def getitem_helper(self, desc, row, for_eii=False):
        if self.cache_image_bytes:
            raise NotImplementedError
            image_bytes = self.image_byte_cache[(desc.path, desc.subject, desc.direction)] # TODO make sure the cache is saving tensors using the right function (seen below)
        else:
            def image_byte_getter(x):
                with open(str(x), "rb") as f:
                    return np.frombuffer(f.read(), dtype=np.uint8)
            image_names = [row['image_names'][f] for f in desc.frames]
            image_bytes = [torchvision.io.read_file(os.path.join(row['images_dir'], name)) for name in image_names] 

        gaze_points = self.gaze_cache[(desc.path, desc.subject, desc.direction)][desc.frames]
        
        if for_eii:
            return image_bytes, gaze_points

        gazes_hm = torch.stack([self.gaussian_generator(g) for g in gaze_points])

        image_list = torchvision.io.decode_jpeg(image_bytes, device=self.decoding_device)
        images =  torch.stack(image_list)
        images = images.float() / 255.0
        images = self.normalize(images)

        return {
            'images': images,
            'gazes': gaze_points,
            'gazes_hm': gazes_hm
        }

    def __len__(self):    
        return self.items_per_video.sum()

class ExternalInputIterator():
    def __init__(self, dataset: EgoCampusDataset, cfg):
        self.dataset = dataset
        mode = self.dataset.mode
        self.batch_size = cfg['data'][mode]['batch_size']
        self.len = (len(dataset) // self.batch_size) * self.batch_size

        # TODO figure out why this is the case
        assert self.dataset.decoding_device == 'cuda', 'The decoding device of the dataset must be cuda for best performance'

        if mode == "train":
            self.idxs = np.random.permutation(self.len)
        else:
            self.idxs = np.arange(self.len)

    def __iter__(self):
        self.counter = 0
        return self

    def __len__(self):
        return self.len

    def __next__(self):
        if self.counter >= self.len:
            self.counter = 0
            raise StopIteration

        image_bytes_batch = []
        gaze_points_batch = []
        for _ in range(self.batch_size):
            idx = self.idxs[self.counter]

            image_bytes, gaze_points = self.dataset.__getitem__(idx, for_eii=True)
            image_bytes_batch.extend(image_bytes)
            gaze_points_batch.extend(gaze_points)
            self.counter += 1

        return image_bytes_batch, gaze_points_batch

class EgoCampusDataLoader():
    def __init__(self, dataset: EgoCampusDataset, cfg):
        self.dataset = dataset
        mode = self.dataset.mode
        self.cfg = cfg
        self.eii = ExternalInputIterator(dataset, cfg)
        self.batch_size = cfg['data'][mode]['batch_size']

        self.imgs_per_item = cfg['data']['context_window'] // cfg['data']['sample_stride']
        pipe = Pipeline(
            batch_size=self.imgs_per_item * self.batch_size,
            num_threads=2,
            device_id=0
        )

        with pipe:
            jpeg_bytes, gaze_points = fn.external_source(
                source=self.eii, num_outputs=2, dtype=[types.UINT8, types.FLOAT]
            )
            decoded = fn.decoders.image(jpeg_bytes, device="mixed", output_type=types.RGB)
            decoded = fn.transpose(decoded, perm=[2,0,1]) / 255.0
            decoded = fn.normalize(decoded, mean=cfg['data']['mean'][0], stddev=cfg['data']['std'][0]) # TODO does dali support list/tensor for mean?
            pipe.set_outputs(decoded, gaze_points)

        pipe.build()
        self.dali_iterator = DALIGenericIterator(pipe, ['images', 'gaze_pts'], last_batch_policy=LastBatchPolicy.DROP, auto_reset=True)

    def __len__(self):
        return len(self.eii) // self.batch_size

    def __iter__(self):
        self.dali_iterator.reset()
        return self

    def __next__(self):
        try:
            item = next(self.dali_iterator)[0]
            images = item['images']
            gaze_pts = item['gaze_pts'].view((self.batch_size, self.imgs_per_item, 2))
            gazes_hm = torch.stack([self.dataset.gaussian_generator(g) for g in gaze_pts[:, -1]])

            return {
                'images': images.view((self.batch_size, self.imgs_per_item) + images.shape[1:]),
                'gazes': gaze_pts[:, -1],
                'gazes_hm': gazes_hm
            }
            
        except StopIteration:
            self.dali_iterator.reset()
            raise

def gaussian_generator_test():
    mask_path = ''
    mask = Image.open(mask_path).convert('L')

    gg = GaussianGenerator((224, 224), sigma=600.0, mask=mask)
    gt = gg(torch.tensor([0.5, 0.5]))

    plt.figure()
    plt.imshow(gt)
    plt.show()

def dataset_example():
    config = utils.get_config()
    dataset = EgoCampusDataset(config, decoding_device='cpu')

    print(f'{len(dataset)=}')
    print(f'{dataset.frames_per_video.sum()=}')

    sample_item = dataset[0]
    images = sample_item['images']
    gazes = sample_item['gazes']
    gazes_hm = sample_item['gazes_hm']

    t, c, h, w = images.shape
    fig, axs = plt.subplots(nrows=2, ncols=t)
    for i in range(t):
        axs[0][i].imshow(images[i].permute(1,2,0))
        axs[0][i].scatter(*gazes[i].T * w, marker='+', s=350, c='red')
        axs[1][i].imshow(gazes_hm[i])

    for ax in axs.flatten():
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def dataloader_example():
    config = utils.get_config()
    dataset = EgoCampusDataset(config, decoding_device='cuda') # decoding device must be cuda for use with loader
    loader = EgoCampusDataLoader(dataset, config)
    loader = iter(loader)
    batch = next(loader)

    print(batch.keys()) # ['images', 'gazes', 'gazes_hm']

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # dataset_example()
    dataloader_example()