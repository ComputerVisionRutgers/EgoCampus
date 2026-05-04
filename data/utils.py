from triton.language.extra import spec
import argparse
import sys
import os
import gc
import time
from functools import reduce
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import psutil
import pyvrs
import tomllib
import ubelt as ub
import torch.nn.functional as F
from projectaria_tools.core import data_provider, calibration
from pathlib import Path
from PIL import Image
from jaxtyping import Float
from turbojpeg import TurboJPEG, TJPF_RGB
from torch import Tensor

RGB_STREAM_ID = "214-1"
ET_STREAM_ID = "211-1"
IMU_LEFT_ID = "1202-2"
IMU_RIGHT_ID = "1202-1"

def load_toml_config(filepath):
    with open(filepath, 'rb') as f:
        return tomllib.load(f)

def load_toml(x):
    return load_toml_config(x)

def truncate_string_end(s, maxlen=30):
    """Truncate the starts of strings"""
    if isinstance(s, ub.Path):
        s = str(s)
    if isinstance(s, str) and len(s) > maxlen:
        return '...' + s[-(maxlen - 3):]
    return s

def display_tail_strings(df, maxlen=30):
    """Display a dataframe, showing the tail ends of strings instead of cutting them off"""
    df_copy = df.map(lambda x: truncate_string_end(x, maxlen))
    print(df_copy)

def get_memory_usage():
    memory_info = psutil.virtual_memory()
    return memory_info.percent

class SuppressCOutput:
    """Context manager to suppress the logging statements from project aria tools
    
    ex:
    with SuppressCOutput():
        provider = data_provider.create_vrs_dataprovider(path)
    """
    def __enter__(self):
        self._original_stdout_fd = os.dup(1)
        self._original_stderr_fd = os.dup(2)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 1)
        os.dup2(self._devnull, 2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self._original_stdout_fd, 1)
        os.dup2(self._original_stderr_fd, 2)
        
        os.close(self._devnull)
        os.close(self._original_stdout_fd)
        os.close(self._original_stderr_fd)
    
def vrs_to_numpy_array(
        vrs_path: Optional[Path | ub.Path] = None,
        scaling_factor: tuple[int, int] = (1,1),
        post_process: Optional[Callable] = None,
        with_timestamp: bool = False,
        timeit: bool = False,
        min: int | None = None,
        max: int | None = None,
    ) -> np.ndarray | tuple[np.ndarray, list]:
    """Load all rgb images from a vrs file into a numpy array

    Parameters:
        vrs_path: path to the vrs file
        scaling_factor: applied during jpeg decoding to efficiency decode image to the specified scale.
          you can't use arbitrary scales. useful options:
            (1,1) for 1408x1408px
            (3,8) for 528x528px
            (1,4) for 352x352px
            (1,8) for 176x176px
        post_process: a function to be applied to the images after being loaded. usually pass the camera
          calibration  function here. post processing functions are applied to each image. therefore, if
          there is a post processing step that can operate on an entire numpy array at once (i.e. rot90),
          you should apply that at the end instead of with `post_process`.
        with_timestamp: a bool to indicate the function should also return timestamps for all the images
        timeit: debug print statement to time how long this takes
        min: minimum index to convert to numpy array
        max: maxmium index to convert to numpy array

    Returns:
        numpy array of shape [num_images, 3, H, W]
    """
    gc.collect()

    if timeit:
        start = time.time()

    sync_reader = pyvrs.reader.SyncVRSReader(vrs_path)
    reader = sync_reader.filtered_by_fields(stream_ids={RGB_STREAM_ID}, record_types={'data'})

    num_rgb_frames = len(reader)

    h = int(1408 * scaling_factor[0] / scaling_factor[1])
    dummy_input = np.empty((h, h, 3))
    if post_process is not None:
        dummy_input = post_process(dummy_input)

    frames = np.empty((num_rgb_frames,)+dummy_input.shape, dtype=np.uint8)

    jpeg = TurboJPEG()
    jpeg_bytes = [None] * num_rgb_frames

    for i in range(num_rgb_frames):
        jpeg_bytes[i] = reader[i].image_blocks[0]

    def _worker(idx, buf):
        if post_process is None:
            jpeg.decode(buf, pixel_format=TJPF_RGB, dst=frames[idx], scaling_factor=scaling_factor)
        else:
            frames[idx] = post_process(jpeg.decode(buf, pixel_format=TJPF_RGB, scaling_factor=scaling_factor))

    print(f'memory usage before threadpool: {get_memory_usage()}')
    with ThreadPoolExecutor(max_workers=1) as ex:
        for idx, buf in enumerate(jpeg_bytes):
           _ = ex.submit(_worker, idx, buf)

    if timeit:
        print(f'vrs_to_numpy_array took {round(time.time() - start, 3)} seconds')

    sync_reader.close()

    if with_timestamp:
        return frames, reader.get_timestamp_list()
    else:
        return frames


def vrs_to_jpeg_bulk(in_dir: ub.Path, out_dir: ub.Path):
    for child in in_dir.ls():
        for folder in child.ls():
            for sub_child in folder.ls("*.vrs"):
                provider = data_provider.create_vrs_data_provider(str(sub_child))
                calib = provider.get_device_calibration().get_camera_calib("camera-rgb")
                pinhole = calibration.get_linear_camera_calibration(512, 512, 150, "camera-rgb", calib.get_transform_device_camera())
                provider = 0
                undistort_image = lambda x: calibration.distort_by_calibration(x, pinhole, calib)
                images = np.rot90(vrs_to_numpy_array(sub_child, post_process=undistort_image), axes=(2,1))

                # Extract the letter before '.vrs' in the filename
                filename = sub_child.name  # 'U_S06_P02_r.vrs'
                letter = filename.split('_')[-1][0]  # get 'r' or 'f'

                # Map letter to folder name
                folder_map = {'f': 'forward', 'r': 'reverse'}
                folder_name = folder_map.get(letter, 'unknown')  # default to 'unknown' if letter unexpected

                # Build new path as before but replace last part with folder_name
                parts = list(sub_child.parts)
                busch_index = parts.index('BuschPaths_Final')

                new_parts = (
                    parts[:busch_index] +
                    ['BuschPaths_Images'] +
                    parts[busch_index + 1:-1] +
                    [folder_name]
                )

                new_path = ub.Path(*new_parts).ensuredir()
            print(new_path)


            for idx, i in enumerate(images):
                img = Image.fromarray(i)
                # img.save(f"{str(new_path)}/{idx}.jpg")  # you can change format by file extension
                print(f'saving image to {new_path}')
            images = []

# TODO actually use type checking
def torch_img_to_np_img(img: Float[Tensor, "... 3 H W"]) -> Float[Tensor, "... H W 3"]:
    return np.moveaxis(img.detach().cpu().numpy(), -3, -1)


def parse_args():
    """
    Args:
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide EgoCampus video training and testing pipeline."
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="./configs/default.toml",
        type=str,
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser.parse_args()

def load_config(args):
    with open(args.cfg_file, 'rb') as f:
        cfg = tomllib.load(f)

    opt_args = {args.opts[i]: args.opts[i+1] for i in range(0, len(args.opts), 2)}

    for keys, v in opt_args.items():
        temp_dict = v
        for k in reversed(keys.split('.')):
            temp_dict = {k: temp_dict}

        cfg = merge_dicts(cfg, temp_dict)

    return cfg

def merge_dicts(a: dict, b: dict, path=[]):
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key], path + [str(key)])
            elif a[key] != b[key]:
                # raise Exception('Conflict at ' + '.'.join(path + [str(key)]))
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a

def frame_softmax(logits, temperature):
    # breakpoint()
    batch_size, H, W = logits.shape
    # reshape -> softmax (dim=-1) -> reshape back
    logits = logits.view(batch_size, H * W)
    atten_map = F.softmax(logits / temperature, dim=-1)
    atten_map = atten_map.view(batch_size, H, W)
    return atten_map


def validate_config(config: dict) -> dict:
    assert set(config['data']['test']['included_paths']).isdisjoint(set(config['data']['train']['included_paths'])), \
        "There should not be overlap between test and train sets"

    return config

def get_config_file(default='./configs/default.toml'):
    if len(sys.argv) > 1 and sys.argv[1] == '--config':
        assert len(sys.argv) > 2, "passed '--config' flag but did not specify a config file" 
        specified_config = sys.argv[2]
    elif 'config' in os.environ:
        specified_config = os.environ['config']
    else:
        return default

    assert specified_config.endswith('.toml'), "config file must be .toml"
    assert os.path.isfile(specified_config), "specified config file does not exist"
    return specified_config
    

def get_config():
    import tyro
    
    config_file = get_config_file(default='./configs/default.toml')
    with open(config_file, 'rb') as f:
        config = tomllib.load(f)
    
    config['config'] = config_file
    overwritten_config = tyro.cli(dict, default=config)
    return validate_config(overwritten_config)

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)

    print(args)
    print(f'{cfg=}')