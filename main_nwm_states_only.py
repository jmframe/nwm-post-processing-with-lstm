"""Docstring"""
import argparse
import json
import pickle
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import KFold

from papercode.datasets import CamelsH5v2, CamelsTXTv2
from papercode.datautils import add_camels_attributes, load_attributes
from papercode.ealstm import EALSTM
from papercode.emblstm import EmbLSTM
from papercode.features import get_additional_features
from papercode.lstm import LSTM
from papercode.metrics import calc_nse
from papercode.nseloss import NSELoss
from papercode.utils import create_h5_files_v2, get_basin_list
from papercode.features import training_period_climate_indices

###########
# Globals #
###########

# fixed settings for all experiments
GLOBAL_SETTINGS = {
    'batch_size': 512,
    'clip_norm': True,
    'clip_value': 1,
    'dropout': 0.4,
    'epochs': 30,
    'hidden_size': 256,
    'initial_forget_gate_bias': 5,
    'log_interval': 50,
    'learning_rate': 1e-3,
    'seq_length': 365,
    #'embedding_hiddens': [35, 30, 30],
    'embedding_hiddens': [],
    'train_start': pd.to_datetime('01102004', format='%d%m%Y'),
    'train_end': pd.to_datetime('30092014', format='%d%m%Y'),
    'val_start': pd.to_datetime('01101994', format='%d%m%Y'),
    'val_end': pd.to_datetime('30092003', format='%d%m%Y'),
    #'val_start': pd.to_datetime('01101989', format='%d%m%Y'),
    #'val_end': pd.to_datetime('30091999', format='%d%m%Y'),
    # list of CAMELS attributes to add to the static inputs
#    'camels_attr': ['elev_mean', 'slope_mean', 'area_gages2', 'carbonate_rocks_frac',
#                    'geol_permeability', 'soil_depth_pelletier', 'soil_depth_statsgo',
#                    'soil_porosity', 'soil_conductivity', 'max_water_content', 'sand_frac',
#                    'silt_frac', 'clay_frac'],
    'camels_attr': ['elev_mean', 'slope_mean', 'area_gages2', 'frac_forest',
                    'lai_max', 'lai_diff', 'gvf_max', 'gvf_diff',
                    'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
                    'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac',
                    'carbonate_rocks_frac', 'geol_permeability'],
    'dynamic_inputs': ['qBtmVertRunoff',
                       'ACCET','FSNO','SNOWH',
                       'SOIL_M1','SOIL_M2','SOIL_M3','SOIL_M4','SOIL_W1','SOIL_W2','SOIL_W3','SOIL_W4',
                       'TRAD','UGDRNOFF',
                       'mean_sfcheadsubrt','mean_zwattablrt','max_sfcheadsubrt','max_zwattablrt'],
    # list of column names to use as additional static inputs
    'static_inputs': []
}

###############
# Prepare run #
###############


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args() -> Dict:
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'mode', 
        choices=["train", "evaluate", "eval_robustness", "create_splits"])

    parser.add_argument(
        '--gpu',
        type=int,
        default=-1,
        help="User-selected GPU ID - if none chosen, will default to cpu")

    parser.add_argument('--camels_root', 
        type=str,
        default='data/basin_dataset_public_v1p2/', 
        help="Root directory of CAMELS data set")

    parser.add_argument('--seed', 
        type=int, 
        required=False, 
        help="Random seed")

    parser.add_argument('--run_dir', 
        type=str, 
        help="For evaluation mode. Path to run directory.")

    parser.add_argument('--cache_data', 
        type=str2bool, 
        default=True, 
        help="If True, loads all data into memory")

    parser.add_argument('--use_dynamic_climate', 
        type=str2bool, 
        default=False, 
        help="Use dynamic (vs. staic) climate indexes?")

    parser.add_argument('--static_climate', 
        type=str, 
        default='user_must_define_value', 
        help="Use train vs. test period static climate stats.")

    parser.add_argument('--num_workers', 
        type=int, 
        default=12, 
        help="Number of parallel threads for data loading")

    parser.add_argument('--concat_static',
        type=str2bool,
        default=True,
        help="If True, train LSTM with static feats concatenated at each time step")

    parser.add_argument('--use_mse',
        type=str2bool,
        default=False,
        help="If True, uses mean squared error as loss function.")

    parser.add_argument('--train_file', 
         type=str, 
         default=None, 
         help="Preprocessed h5 file to use for training")

    parser.add_argument('--scaler_file', 
         type=str, 
         default=None, 
         help="Pickle file containing the scaler, when preprocessed h5 file is used")

    parser.add_argument('--split_file',
        type=str,
        default=None,
        help="Path to file created from the `create_splits` function.")

    parser.add_argument('--split',
        type=int,
        default=None,
        help="Defines split to use for training/testing in kFold cross validation")

    parser.add_argument('--n_splits',
        type=int,
        default=None,
        help="Number of splits to create for cross validation")

    cfg = vars(parser.parse_args())

    # Validation checks
    if (cfg["mode"] == "train", cfg["mode"] == "create_splits") and (cfg["seed"] is None):
        # generate random seed for this run
        cfg["seed"] = int(np.random.uniform(low=0, high=1e6))

    if (cfg["mode"] in ["evaluate", "eval_robustness"]) and (cfg["run_dir"] is None):
        raise ValueError("In evaluation mode a run directory (--run_dir) has to be specified")

    if (cfg["train_file"] is not None) and (cfg["scaler_file"] is None):
        raise ValueError("You have to specify a `scaler_file`, if a `train_file` is passed.")

    # GPU selection
    if cfg["gpu"] >= 0:
      device = f"cuda:{cfg['gpu']}"
    else:
      device = 'cpu'
    global DEVICE
    DEVICE = torch.device(device if torch.cuda.is_available() else "cpu")

    # combine global settings with user config
    cfg.update(GLOBAL_SETTINGS)

    if cfg["mode"] == "train":
        # print config to terminal
        for key, val in cfg.items():
            print(f"{key}: {val}")

    # convert path to PosixPath object
    cfg["camels_root"] = Path(cfg["camels_root"])
    if cfg["run_dir"] is not None:
        cfg["run_dir"] = Path(cfg["run_dir"])
    return cfg


def _setup_run(cfg: Dict) -> Dict:
    """Create folder structure for this run

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config

    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    now = datetime.now()
    day = f"{now.day}".zfill(2)
    month = f"{now.month}".zfill(2)
    hour = f"{now.hour}".zfill(2)
    minute = f"{now.minute}".zfill(2)
    second = f"{now.second}".zfill(2)

    if cfg['split_file'] == None: 
       run_name = f'run_{day}{month}_{hour}{minute}{second}_seed{cfg["seed"]}'
    else:
       run_name = f'run_{day}{month}_{hour}{minute}{second}_seed{cfg["seed"]}_split{cfg["split"]}'

    cfg['run_dir'] = Path(__file__).absolute().parent / "runs" / run_name

    if not cfg["run_dir"].is_dir():
        cfg["train_dir"] = cfg["run_dir"] / 'data' / 'train'
        cfg["train_dir"].mkdir(parents=True)
        cfg["val_dir"] = cfg["run_dir"] / 'data' / 'val'
        cfg["val_dir"].mkdir(parents=True)
    else:
        raise RuntimeError(f"There is already a folder at {cfg['run_dir']}")

    # dump a copy of cfg to run directory
    with (cfg["run_dir"] / 'cfg.json').open('w') as fp:
        temp_cfg = {}
        for key, val in cfg.items():
            if isinstance(val, PosixPath):
                temp_cfg[key] = str(val)
            elif isinstance(val, pd.Timestamp):
                temp_cfg[key] = val.strftime(format="%d%m%Y")
            else:
                temp_cfg[key] = val
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)

    return cfg


def _prepare_data(cfg: Dict, basins: List) -> Dict:
    """Preprocess training data.

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config
    basins : List
        List containing the 8-digit USGS gauge id

    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    # create database file containing the static basin attributes
    cfg["db_path"] = str(cfg["run_dir"] / "attributes.db")
    add_camels_attributes(cfg["camels_root"], db_path=cfg["db_path"])

    # create .h5 files for train and validation data
    if cfg["train_file"] is None:
        cfg["train_file"] = cfg["train_dir"] / 'train_data.h5'
        cfg["scaler_file"] = cfg["train_dir"] / "scaler.p"

        # get additional static inputs
        file_name = Path(__file__).parent / 'data' / 'dynamic_features_nwm_v2.p'
        with file_name.open("rb") as fp:
            additional_features = pickle.load(fp)
        # ad hoc static climate indices 
        # requres the training period for this experiment
        # overwrites the *_dyn type climate indices in 'additional_features'
        if not cfg['use_dynamic_climate']:
            train_clim_indexes = training_period_climate_indices(
                                      db_path=cfg['db_path'], camels_root=cfg['camels_root'],
                                      basins=basins, 
                                      start_date=cfg['train_start'], end_date=cfg['train_end'])
            for basin in basins:
               for col in train_clim_indexes[basin].columns:
                   additional_features[basin][col] = np.tile(train_clim_indexes[basin][col].values,[additional_features[basin].shape[0],1])

        create_h5_files_v2(
            camels_root=cfg["camels_root"],
            out_file=cfg["train_file"],
            basins=basins,
            dates=[cfg["train_start"], cfg["train_end"]],
            db_path=cfg["db_path"],
            cfg=cfg,
            additional_features=additional_features,
            num_workers=cfg["num_workers"],
            seq_length=cfg["seq_length"])
    
    # copy scaler file into run folder
    else:
        dst = cfg["train_dir"] / "scaler.p"
        shutil.copyfile(cfg["scaler_file"], dst)

    return cfg


################
# Define Model #
################


class Model(nn.Module):
    """Wrapper class that connects LSTM/EA-LSTM with fully connceted layer"""

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 embedding_hiddens: List = [],
                 initial_forget_bias: int = 5,
                 dropout: float = 0.0,
                 concat_static: bool = True):
        """Initialize model.

        Parameters
        ----------
        input_size_dyn: int
            Number of dynamic input features.
        input_size_stat: int
            Number of static input features (used in the EA-LSTM input gate).
        hidden_size: int
            Number of LSTM cells/hidden units.
        initial_forget_bias: int
            Value of the initial forget gate bias. (default: 5)
        dropout: float
            Dropout probability in range(0,1). (default: 0.0)
        concat_static: bool
            If True, uses standard LSTM otherwise uses EA-LSTM
        """
        super(Model, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.embedding_hiddens = embedding_hiddens
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout
        self.concat_static = concat_static

        if self.concat_static:
            if self.embedding_hiddens:
                self.lstm = EmbLSTM(input_size_dyn=input_size_dyn, 
                                    input_size_stat=input_size_stat,
                                    hidden_size=hidden_size, 
                                    embedding_hiddens=embedding_hiddens,
                                    initial_forget_bias=initial_forget_bias)
            else:
                self.lstm = LSTM(
                    input_size=input_size_dyn,
                    hidden_size=hidden_size,
                    initial_forget_bias=initial_forget_bias)
        else:
            self.lstm = EALSTM(
                input_size_dyn=input_size_dyn,
                input_size_stat=input_size_stat,
                hidden_size=hidden_size,
                initial_forget_bias=initial_forget_bias)

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass through the model.

        Parameters
        ----------
        x_d : torch.Tensor
            Tensor containing the dynamic input features of shape [batch, seq_length, n_features]
        x_s : torch.Tensor, optional
            Tensor containing the static catchment characteristics, by default None

        Returns
        -------
        out : torch.Tensor
            Tensor containing the network predictions
        h_n : torch.Tensor
            Tensor containing the hidden states of each time step
        c_n : torch,Tensor
            Tensor containing the cell states of each time step
        """
        if x_s is None:
            h_t, c_t = self.lstm(x_d)
            c_n = torch.squeeze(c_t[:,-1,:])
            e_n = torch.zeros(c_t.shape[0])
        else:
            h_t, (h_n, c_n), e_n = self.lstm(x_d, x_s)
            e_n = torch.transpose(e_n, 1, 0)[-1,:,:]
            c_n = torch.squeeze(c_n)

        last_h = self.dropout(h_t[:, -1, :])
        out = self.fc(last_h)

        return out, c_n, e_n 


###########################
# Train or evaluate model #
###########################


def train(cfg):
    """Train model.

    Parameters
    ----------
    cfg : Dict
        Dictionary containing the run config
    """
    # fix random seeds
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    if cfg["split_file"] is not None:
        with Path(cfg["split_file"]).open('rb') as fp:
            splits = pickle.load(fp)
        basins = splits[cfg["split"]]["train"]
    else:
        basins = get_basin_list()
        #basins = basins[:30]

    # create folder structure for this run
    cfg = _setup_run(cfg)

    # prepare data for training
    cfg = _prepare_data(cfg=cfg, basins=basins)

    with open(cfg["scaler_file"], 'rb') as fp:
        scaler = pickle.load(fp)

    camels_attr = load_attributes(cfg["db_path"], basins, drop_lat_lon=True, 
                                  keep_features=cfg["camels_attr"])
    scaler["camels_attr_mean"] = camels_attr.mean()
    scaler["camels_attr_std"] = camels_attr.std()

    # create model and optimizer
    if cfg["concat_static"] and not cfg["embedding_hiddens"]:
        input_size_stat = 0
        input_size_dyn = (len(cfg["dynamic_inputs"]) + 
                          len(cfg["camels_attr"]) + 
                          len(cfg["static_inputs"]))
        concat_static = True
    else:
        input_size_stat = len(cfg["camels_attr"]) + len(cfg["static_inputs"])
        input_size_dyn = len(cfg["dynamic_inputs"])
        concat_static = False
    model = Model(
        input_size_dyn=input_size_dyn,
        input_size_stat=input_size_stat,
        hidden_size=cfg["hidden_size"],
        initial_forget_bias=cfg["initial_forget_gate_bias"],
        embedding_hiddens=cfg["embedding_hiddens"],
        dropout=cfg["dropout"],
        concat_static=cfg["concat_static"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    # prepare PyTorch DataLoader
    ds = CamelsH5v2(
        h5_file=cfg["train_file"],
        basins=basins,
        db_path=cfg["db_path"],
        concat_static=concat_static,
        cache=cfg["cache_data"],
        camels_attr=cfg["camels_attr"],
        scaler=scaler)
    loader = DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])

    # define loss function
    if cfg["use_mse"]:
        loss_func = nn.MSELoss()
    else:
        loss_func = NSELoss()

    # reduce learning rates after each 10 epochs
    learning_rates = {11: 5e-4, 21: 1e-4}

    for epoch in range(1, cfg["epochs"] + 1):
        # set new learning rate
        if epoch in learning_rates.keys():
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rates[epoch]

        train_epoch(model, optimizer, loss_func, loader, cfg, epoch, cfg["use_mse"])

        model_path = cfg["run_dir"] / f"model_epoch{epoch}.pt"
        torch.save(model.state_dict(), str(model_path))


def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, loss_func: nn.Module,
                loader: DataLoader, cfg: Dict, epoch: int, use_mse: bool):
    print('train_epoch')
    """Train model for a single epoch.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    optimizer : torch.optim.Optimizer
        Optimizer used for weight updating
    loss_func : nn.Module
        The loss function, implemented as a PyTorch Module
    loader : DataLoader
        PyTorch DataLoader containing the training data in batches.
    cfg : Dict
        Dictionary containing the run config
    epoch : int
        Current Number of epoch
    use_mse : bool
        If True, loss_func is nn.MSELoss(), else NSELoss() which expects addtional std of discharge
        vector

    """
    model.train()

    # process bar handle
    pbar = tqdm(loader, file=sys.stdout)
    pbar.set_description(f'# Epoch {epoch}')

    # Iterate in batches over training set
    for data in pbar:
#        print('\n')
        # delete old gradients
        optimizer.zero_grad()

        # forward pass through LSTM
        if len(data) == 3:
            x, y, q_stds = data
            x, y, q_stds = x.to(DEVICE), y.to(DEVICE), q_stds.to(DEVICE)
            predictions = model(x)[0]

        # forward pass through EALSTM
        elif len(data) == 4:
            x_d, x_s, y, q_stds = data
            x_d, x_s, y = x_d.to(DEVICE), x_s.to(DEVICE), y.to(DEVICE)
            predictions = model(x_d, x_s)[0]

        # MSELoss
        mask = ~torch.isnan(predictions) 
        if use_mse:
            loss = loss_func(predictions[mask], y[mask])

        # NSELoss needs std of each basin for each sample
        else:
            q_stds = q_stds.to(DEVICE)
            loss = loss_func(predictions[mask], y[mask], q_stds)

        # calculate gradients
        loss.backward()

        if cfg["clip_norm"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_value"])

        # perform parameter update
        optimizer.step()

        pbar.set_postfix_str(f"Loss: {loss.item():5f}")


def evaluate(user_cfg: Dict):
    """Train model for a single epoch.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
        
    """
    with open(user_cfg["run_dir"] / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)

    if user_cfg["split_file"] is not None:
        with Path(user_cfg["split_file"]).open('rb') as fp:
            splits = pickle.load(fp)
        basins = splits[run_cfg["split"]]["test"]
    else:
        basins = get_basin_list()

    # get attribute means/stds
    db_path = str(user_cfg["run_dir"] / "attributes.db")
    attributes = load_attributes(db_path=db_path, basins=basins, drop_lat_lon=True,
                                 keep_features=user_cfg["camels_attr"])

    # get remaining scaler from pickle file
    scaler_file = user_cfg["run_dir"] / "data" / "train" / "scaler.p"
    with open(scaler_file, "rb") as fp:
        scaler = pickle.load(fp)
    scaler["camels_attr_mean"] = attributes.mean()
    scaler["camels_attr_std"] = attributes.std()

    # create model
    if run_cfg["concat_static"] and not run_cfg["embedding_hiddens"]:
        input_size_stat = 0
        input_size_dyn = (len(run_cfg["dynamic_inputs"]) + 
                          len(run_cfg["camels_attr"]) + 
                          len(run_cfg["static_inputs"]))
        concat_static = True
    else:
        input_size_stat = len(run_cfg["camels_attr"]) + len(run_cfg["static_inputs"])
        input_size_dyn = len(run_cfg["dynamic_inputs"])
        concat_static = False
    model = Model(
        input_size_dyn=input_size_dyn,
        input_size_stat=input_size_stat,
        hidden_size=run_cfg["hidden_size"],
        dropout=run_cfg["dropout"],
        concat_static=run_cfg["concat_static"],
        embedding_hiddens=run_cfg["embedding_hiddens"]).to(DEVICE)

    # load trained model
    weight_file = user_cfg["run_dir"] / 'model_epoch30.pt' 
    model.load_state_dict(torch.load(weight_file, map_location=DEVICE))

    date_range = pd.date_range(start=user_cfg["val_start"], end=user_cfg["val_end"])
    results = {}
    cell_states = {}
    embeddings = {}
    nses = []

    file_name = Path(__file__).parent / 'data' / 'dynamic_features_nwm_v2.p'
    with file_name.open("rb") as fp:
        additional_features = pickle.load(fp)
    
    # ad hoc static climate indices 
    # requres the training period for this experiment
    # overwrites the *_dyn type climate indices in 'additional_features'
    if not user_cfg['use_dynamic_climate']:
        if user_cfg['static_climate'].lower() == 'test':
            eval_clim_indexes = training_period_climate_indices(
                                  db_path=db_path, camels_root=user_cfg['camels_root'],
                                  basins=basins, 
                                  start_date=user_cfg['val_start'], end_date=user_cfg['val_end'])
        elif user_cfg['static_climate'].lower() == 'train':
            eval_clim_indexes = training_period_climate_indices(
                                  db_path=db_path, camels_root=user_cfg['camels_root'],
                                  basins=basins, 
                                  start_date=user_cfg['train_start'], end_date=user_cfg['train_end'])
        else:
            raise RuntimeError(f"Unknown static_climate variable.")
            

        for basin in basins:
           for col in eval_clim_indexes[basin].columns:
               additional_features[basin][col] = np.tile(eval_clim_indexes[basin][col].values,[additional_features[basin].shape[0],1])

    for basin in tqdm(basins):
        ds_test = CamelsTXTv2(
            camels_root=user_cfg["camels_root"],
            basin=basin,
            dates=[user_cfg["val_start"], user_cfg["val_end"]],
            is_train=False,
            dynamic_inputs=user_cfg["dynamic_inputs"],
            camels_attr=user_cfg["camels_attr"],
            static_inputs=user_cfg["static_inputs"],
            additional_features=additional_features[basin],
            scaler=scaler,
            seq_length=run_cfg["seq_length"],
            concat_static=concat_static,
            db_path=db_path)
        loader = DataLoader(ds_test, batch_size=2500, shuffle=False, 
                            num_workers=user_cfg["num_workers"])

        preds, obs, cells, embeds = evaluate_basin(model, loader)

        # rescale predictions
        preds = preds * scaler["q_std"] + scaler["q_mean"]

        # store predictions
        # set discharges < 0 to zero
        preds[preds < 0] = 0
        nses.append(calc_nse(obs[obs >= 0], preds[obs >= 0]))
        df = pd.DataFrame(data={'qobs': obs.flatten(), 'qsim': preds.flatten()}, index=date_range)
        results[basin] = df

        # store cell states and embedding values
        cell_states[basin] = pd.DataFrame(data=cells, index=date_range)
        embeddings[basin] = pd.DataFrame(data=embeds, index=date_range)
        
    print(f"Mean NSE {np.mean(nses)}, median NSE {np.median(nses)}")

    _store_results(user_cfg, run_cfg, results, cell_states, embeddings)


def evaluate_basin(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate model on a single basin

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    loader : DataLoader
        PyTorch DataLoader containing the basin data in batches.

    Returns
    -------
    preds : np.ndarray
        Array containing the (rescaled) network prediction for the entire data period
    obs : np.ndarray
        Array containing the observed discharge for the entire data period

    """
    model.eval()

    preds, obs = None, None
    cells, embeds = None, None

    with torch.no_grad():
        for data in loader:
            if len(data) == 2:
                x, y = data
                x, y = x.to(DEVICE), y.to(DEVICE)
                p, c, e = model(x)
            elif len(data) == 3:
                x_d, x_s, y = data
                x_d, x_s, y = x_d.to(DEVICE), x_s.to(DEVICE), y.to(DEVICE)
                p, c, e = model(x_d, x_s)

            if preds is None:
                preds = p.detach().cpu()
                obs = y.detach().cpu()
                cells = c.detach().cpu()
                embeds = e.detach().cpu()
            else:
                preds = torch.cat((preds, p.detach().cpu()), 0)
                obs = torch.cat((obs, y.detach().cpu()), 0)
                cells = torch.cat((cells, c.detach().cpu()), 0)
                embeds = torch.cat((embeds, e.detach().cpu()) ,0)

        preds = preds.numpy()
        obs = obs.numpy()
        cells = cells.numpy()
        embeds = embeds.numpy()

    return preds, obs, cells, embeds


def _store_results(user_cfg: Dict, run_cfg: Dict, results: pd.DataFrame, cells: pd.DataFrame, embeddings: pd.DataFrame):
    print('store_result')
    """Store results in a pickle file.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
    run_cfg : Dict
        Dictionary containing the run config loaded from the cfg.json file
    results : pd.DataFrame
        DataFrame containing the observed and predicted discharge.

    """

    # save lstm predictions
    file_name = user_cfg["run_dir"] / f"lstm_seed{run_cfg['seed']}"
    if user_cfg['use_dynamic_climate']: 
        file_name = f"{file_name}_dynclim"
    else:  
        file_name = f"{file_name}_statclim{user_cfg['static_climate'].lower()}"

    if user_cfg['split_file'] is not None: 
        file_name = f"{file_name}_{user_cfg['split']}"

    file_name = f"{file_name}.p"

    with open(file_name, 'wb') as fp:
        pickle.dump(results, fp)
    print(f"Sucessfully store predictions and observations at {file_name}")

    # save lstm cell states
    file_name = user_cfg["run_dir"] / f"lstm_cell_states_seed{run_cfg['seed']}"
    if user_cfg['use_dynamic_climate']: 
        file_name = f"{file_name}_dynclim"
    else:  
        file_name = f"{file_name}_statclim{user_cfg['static_climate'].lower()}"

    if user_cfg['split_file'] is not None: 
        file_name = f"{file_name}_{user_cfg['split']}"

    file_name = f"{file_name}.p"

    with open(file_name, 'wb') as fp:
        pickle.dump(cells, fp)
    print(f"Sucessfully store states at {file_name}")

    # save lstm embedding values
    if run_cfg["embedding_hiddens"]:
        file_name = user_cfg["run_dir"] / f"lstm_embeddings_seed{run_cfg['seed']}"
        if user_cfg['use_dynamic_climate']: 
            file_name = f"{file_name}_dynclim"
        else:  
            file_name = f"{file_name}_statclim{user_cfg['static_climate'].lower()}"

        if user_cfg['split_file'] is not None: 
            file_name = f"{file_name}_{user_cfg['split']}"

        file_name = f"{file_name}.p"

        with open(file_name, 'wb') as fp:
            pickle.dump(embeddings, fp)
        print(f"Sucessfully store embeddings at {file_name}")


####################
# Cross Validation #
####################

def create_splits(cfg: dict):
    print('create_splits')
    """Create random k-Fold cross validation splits.
    
    Takes a set of basins and randomly creates n splits. The result is stored into a dictionary,
    that contains for each split one key that contains a `train` and a `test` key, which contain
    the list of train and test basins.
    Parameters
    ----------
    cfg : dict
        Dictionary containing the user entered evaluation config
    
    Raises
    ------
    RuntimeError
        If file for the same random seed already exists.
    FileNotFoundError
        If the user defined basin list path does not exist.
    """
    output_file = (Path(__file__).absolute().parent / f'data/kfolds/kfold_splits_seed{cfg["seed"]}.p')
    # check if split file already already exists
    if output_file.is_file():
        raise RuntimeError(f"File '{output_file}' already exists.")

    # set random seed for reproducibility
    np.random.seed(cfg["seed"])

    # read in basin file
    basins = get_basin_list()

    # create folds
    kfold = KFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])
    kfold.get_n_splits(basins)

    # dict to store the results of all folds
    splits = defaultdict(dict)

    for split, (train_idx, test_idx) in enumerate(kfold.split(basins)):
        # further split train_idx into train/val idx into train and val set

        train_basins = [basins[i] for i in train_idx]
        test_basins = [basins[i] for i in test_idx]

        splits[split] = {'train': train_basins, 'test': test_basins}

    with output_file.open('wb') as fp:
        pickle.dump(splits, fp)

    print(f"Stored dictionary with basin splits at {output_file}")


if __name__ == "__main__":
    config = get_args()
    globals()[config["mode"]](config)
