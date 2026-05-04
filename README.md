# EgoCampus
This repo contains code to parse and load the dataset.

## Dataset Information
**Download the dataset [HERE](https://huggingface.co/datasets/ayetida/egocampus)**

The dataset has the following file structure
- EgoCampus
 - Path2
 - Path3
 - ...
 - Path 26
   - SubjectX
      - forward
      - reverse
        - gaze.npy
        - imu.npy
        - images
          - x.jpg

## Installation
Tested on Ubuntu with Python 3.12
1. (recommended) Create a virtual environment `conda create -n egocampus python=3.12`
2. Install PyTorch and Torchvision ([link](https://pytorch.org/get-started/locally/)). GPU support is optional
3. Install other requirements `pip install -r requirements.txt`

## Usage Notes
`EgoCampusDataset` inherits from PyTorch's `Dataset` class, however we do not reccomend using it with PyTorch's `Dataloader`. We instead provide `EgoCampusDataLoader` which uses DALI to parallelize dataloading on NVIDIA GPUs.