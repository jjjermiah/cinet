import torch
from cinet import *
import numpy as np
import os
import pandas as pd
import json
import sys
from io import StringIO
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pickle
import numpy

# Set the training data directory
file_dir = r'/home/gputwo/bhklab/kevint/cinet/train_data/'
file_list = os.listdir(file_dir)

# Set the parameter grid
# param_grid = { "delta" : [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2] }
param_grid = { "delta" : [0.0, 0.01] }

# Set the directory to which model checkpoint files (.ckpt) will be saved
save_dir = './'

# Remember to choose deepCINET or ECINET below in the GridSearchCV code 

#### PERFORM H-PARAM TUNING #### 
# All the data will be saved to the dict called "data"
data = {}
for file in file_list[:2]: 
    name = file.replace('_response.csv','').replace('rnaseq_','').replace('gene_', '')
    df = pd.read_csv(file_dir + file).set_index('cell_line')
    X = df.iloc[:,1:]
    y = df.iloc[:,0]
    grid = GridSearchCV(deepCINET(modelPath= (save_dir + name + '.ckpt'), device='cpu', batch_size=2**12), param_grid, refit = True, verbose = 3,n_jobs=3)
    grid.fit(X,y)
    data[name] = {
        "best_params" : grid.best_params_,
        "cv_results" : grid.cv_results_,
    }

# Modify the content in "data" so that numpy ndarrays and numpy maskedArrays are converted to lists
# This is required so that the results can be saved into json

for key in data:
    for result in data[key]:
        for x in data[key][result]:
            dataType = type(data[key][result][x])
            if dataType == numpy.ndarray: 
                data[key][result][x] = data[key][result][x].tolist()
            elif dataType == numpy.ma.core.MaskedArray:
                data[key][result][x] = data[key][result][x].tolist()

# Save the results into a json file

with open('hparam_tuning.json', 'w') as fp:
    json.dump(data, fp)