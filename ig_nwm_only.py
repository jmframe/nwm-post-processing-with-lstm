#!/att/nobackup/jframe/programs/anaconda3/bin/python3
# coding: utf-8
# Imports
import json
import numpy as np
import pandas as pd
import pickle as pkl
import os
import sys
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm

MAIN_DIR = "/att/nobackup/jframe/lstm_camels/"

# Path to the main CAMELS folder
CAMELS_DIR = MAIN_DIR

# Path to the main directory of this repository
BASE_CODE_DIR = MAIN_DIR + 'papercode/'

# Needed if no precomputed results are used. Path to a single run
#BASE_RUN_DIR = MAIN_DIR+'runs/run_2003_191127_seed83357'
BASE_RUN_DIR = MAIN_DIR+'runs/run_2306_104142_seed83357' # Updated June 22th

sys.path.append(BASE_CODE_DIR)
from main_nwmlstm import Model
from papercode.utils import get_basin_list
from papercode.datautils import load_attributes
from papercode.datasets import CamelsTXTv2

precip_location_in_array = 0

# Use GPU if available
use_cuda = torch.cuda.is_available()
print('CUDA available?', use_cuda)
DEVICE = torch.device("cuda:0" if use_cuda else exit())#"cpu")
print('using device', DEVICE)

# Start and end date of the validation period
VAL_START = pd.to_datetime('01101994', format='%d%m%Y')
VAL_END = pd.to_datetime('30092003', format='%d%m%Y')

# Convert to PosixPaths
CAMELS_DIR = Path(CAMELS_DIR)
BASE_RUN_DIR = Path(BASE_RUN_DIR)

# load run config
with open(BASE_RUN_DIR / "cfg.json", "r") as fp:
    cfg = json.load(fp)
cfg

# get list of modeled basins
basins = get_basin_list()
nbasins = len(basins)

# load means/stds from training period
attributes = load_attributes(db_path=str(BASE_RUN_DIR / "attributes.db"), 
                             basins=basins,
                             keep_features=cfg["camels_attr"],
                             drop_lat_lon=True)

# Initialize new model
print('initializing model ...')
model = Model(input_size_dyn=51,
              input_size_stat=0,
              hidden_size=cfg["hidden_size"],
              dropout=cfg["dropout"]).to(DEVICE)
print('finished initializing model')

# load pre-trained weights
print('loading model ...')
weight_file = BASE_RUN_DIR / "model_epoch30.pt"
model.load_state_dict(torch.load(weight_file, map_location=DEVICE))
print('finished loading model')

# load scaler
print('loading scaler ...')
with open("hdf_files/scaler_nwm_only_v2.p", 'rb') as f:
    scaler = pkl.load(f)
scaler["camels_attr_mean"] = attributes.mean()
scaler["camels_attr_std"] = attributes.std()
print('finished loading scaler')

# get additional static inputs
file_name = Path(MAIN_DIR) / 'data' / 'dynamic_features_nwm_only_v2.p'
with file_name.open("rb") as fp:
    additional_features = pkl.load(fp)
additional_features[basins[0]].columns

def _generate_path_input(baseline, x_d, device, m):
    xi = torch.zeros((m, x_d.shape[1], x_d.shape[2]), requires_grad=True).to(device)
    for k in range(m):
        xi[k, :, :] = baseline + k/(m-1)*(x_d - baseline)
    return xi

def integrated_gradients(model, x_d, device, baseline=None, m=1000):
    
    # this is critical each time
    model.zero_grad()

    if baseline is None:
        baseline = x_d.new_zeros(x_d.shape)
        basin_precip = x_d[:,:,precip_location_in_array]
        precip_baseline = torch.Tensor(basin_precip.shape)
        precip_baseline = precip_baseline.fill_(-torch.mean(basin_precip)/torch.std(basin_precip))
        baseline[:,:,precip_location_in_array] = precip_baseline
    else:
        assert baseline.size() == x_d.size(), "Tensor sizes don't match"

    xi = _generate_path_input(baseline, x_d, device, m)
    xi = torch.autograd.Variable(xi, requires_grad=True).to(device)
    
    out = model(xi)[0]

    out.backward(out.new_ones(m, 1))
    
    path_integral = (1/m) * xi.grad.sum(0)

    igrad = path_integral * (x_d - baseline)

    # difference between baseline and target input
    quality = (out[0, 0] - out[-1, 0]).detach().cpu().numpy()

    return igrad.detach().cpu().numpy(), quality, out[-1,0].detach().cpu().numpy() 

random_basins = basins
random.shuffle(random_basins)
b=0
for basin in random_basins:
    fname = f'igs/{basin}_precip.p'
    basin_ig_file_exists = os.path.exists(fname)
    if basin_ig_file_exists:
        continue
    else:
        print('calculating integrated gradients (precip mean as base) for basin', basin)
        ds_test = CamelsTXTv2(
            camels_root=Path(cfg["camels_root"]),
            basin=basin,
            dates=[VAL_START, VAL_END],
            is_train=False,
            dynamic_inputs=cfg["dynamic_inputs"],
            camels_attr=cfg["camels_attr"],
            static_inputs=cfg["static_inputs"],
            additional_features=additional_features[basin],
            scaler=scaler,
            seq_length=cfg["seq_length"],
            concat_static = cfg['concat_static'],
            db_path=str(BASE_RUN_DIR / "attributes.db"))
         
        loader = DataLoader(ds_test, batch_size=1)
            
        basin_igs = None
        for i, (x_d, y) in enumerate(loader):
            if basin_igs is None:
                basin_igs = np.full([len(loader),x_d.shape[1],x_d.shape[2]], np.nan)
            x_d = x_d.to(DEVICE)
            basin_igs[i,:], _, _ = integrated_gradients(model, x_d, DEVICE, baseline=None, m=100)
                
        with open(fname,'wb') as f:
            pkl.dump(basin_igs, f)
        b += 1
    if b > 100:
        break
