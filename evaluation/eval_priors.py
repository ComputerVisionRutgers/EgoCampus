import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import gc
import os
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from typing import Optional
from datetime import datetime

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# import jax
# import jax.numpy as jnp
# import jax.dlpack
import numpy as np
import torch.nn.functional as F
import ubelt as ub
from jaxtyping import Float
from torchvision import transforms
from torch import Tensor
from torchvision.transforms import v2
from torch.utils.data import Dataset, Subset, DataLoader, Sampler, default_collate
from tqdm import tqdm

import data.utils
import metrics
from data.datasets import (EgoCampusDataset, EgoCampusDataLoader, ClipDesc, Direction)
from data.utils import parse_args, load_config

# from eval_stfusion import calc_metrics
from PIL import Image

@torch.no_grad()
def quantitative():
    args = parse_args()
    cfg = load_config(args)
    
    test_dataset = EgoCampusDatasetV4(cfg, 'test', decoding_device='cuda')
    test_loader = EgoCampusDataLoaderV4(test_dataset, cfg)

    prior_path = 'checkpoints/center_prior_s100.pt'
    prior = torch.from_numpy(torch.load(prior_path, weights_only=False)).unsqueeze(0)
    prior = prior.repeat(cfg['test']['batch_size'], 1, 1)
    model = lambda _: prior

    print(f'{prior.shape=}')

    metrics = {
    }

    i = 0
    for batch in tqdm(test_loader, desc='calculating perf metrics'):
        images = batch['images']

        preds = model(images).cpu().numpy()
        gazes = batch['gazes'][:, -1].cpu().numpy()
        gazes_hm = batch['gazes_hm'][:, -1].cpu().numpy()

        res = calc_metrics(preds, gazes, gazes_hm)
        for k, v in res.items():
            if not k in metrics:
                metrics[k] = []
            metrics[k].append(v)

        # i += 1
        # if i >= 30:
        #     break

    print(f'Evaluation of center prior with {prior_path}:')
    for k, v in metrics.items():
        print(f'{k}: {np.array(v).mean()}')


if __name__ == '__main__':
    quantitative()

    # prior_path = 'checkpoints/center_prior_60s.pt'
    # prior = torch.from_numpy(
    #     torch.load(prior_path, weights_only=False)
    # ).unsqueeze(0)

    # plt.figure()
    # plt.imshow(prior[0])
    # plt.show()