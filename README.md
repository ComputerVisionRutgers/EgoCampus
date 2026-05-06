# EgoCampus
This repo contains code to parse and load the dataset.

## Dataset Information
**Download the dataset [HERE](https://www.kaggle.com/datasets/ayetida/egocampus512)** Extract .zip before using.

The dataset has the following file structure
- EgoCampus
 - Path{1..26}
   - Subject{1..82}
      - forward
        - ...
      - reverse
        - gaze.npy
        - imu.npy
        - images
          - {0..999}.jpg

### Test/Train Split
Test and train splits are defined in a configuration file. The splits used in the paper are described in `./configs/default.toml`. These configuration files are used when instantiating the dataset to provide the different splits.

## Installation
Tested on Ubuntu with Python 3.12
1. Create a virtual environment `conda create -n egocampus python=3.12`
2. Install PyTorch and Torchvision ([link](https://pytorch.org/get-started/locally/)). GPU support is optional
3. Install Nvidia DALI ([link](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html))
4. Install `pyturbojpeg`: `conda install conda-forge::pyturbojpeg`
5. Install other requirements `pip install -r requirements.txt`

## Usage Notes
Modify the configuation .toml to point towards your dataset directory, or overwrite any of the configuration variables with flags. Run `python -m data.datasets --help` to load the dataset and see all available configuration variables.

`EgoCampusDataset` inherits from PyTorch's `Dataset` class, however we do not reccomend using it with PyTorch's `Dataloader`. We instead provide `EgoCampusDataLoader` which uses DALI to parallelize dataloading on NVIDIA GPUs.
