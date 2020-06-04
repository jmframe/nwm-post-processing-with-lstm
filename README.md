# This is code to post process the National Water Model output with Long Short Term Memory networks.
## The LSTM codebase is copied from `https://github.com/kratzert/ealstm_regional_modeling` (Old codebase)
## Also running the LSTM with the new codebase here: `https://github.com/kratzert/lstm_based_hydrology.git` (New codebase)
## Content of main repository
### `main*.py` Main python file used for training and evaluating of our models, as well as to perform the robustness analysis using the `ealstm_regional_modeling` codebase.
### data/ contains the list of basins (USGS gauge ids) considered in our study
### `jupyter_notes/` working notebooks calculating and plotting the results of our study. These notebooks will be updated and when in their final versions will be your starting point for duplicating results.
### `environment_gpu.yml` can be used with Anaconda or Miniconda to create an environment with all packages needed.
### configs/ Configuration files for running the LSTM with the newer codebase `(lstm_based_hydrology)`
### `ig_nwm_lstm.py` This script is used to calculate the integrated gradients of the LSTM runs from the old codebase. The 'baselines' are all set to zero.
### `ig_nwm_lstm_precip.py` This is the same as above, but this particular calculation sets the baseline of the precipitation to -mean/stdev, leaving the other baselines at zero.
## These experiments are trained on water years 2004 - 2013 and tested on water years 1994 - 2002.
##
