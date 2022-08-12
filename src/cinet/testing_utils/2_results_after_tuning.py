# TRAIN TEST
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

# Set the train and test directories, and their names
train_dir = r'/home/gputwo/bhklab/kevint/cinet/train_data/'

test_directories = {
    "CCLE" : r'/home/gputwo/bhklab/kevint/cinet/train_data/',
    "gCSI" : r'/home/gputwo/bhklab/kevint/cinet/test_data/gCSI_Test_Data/', 
    "GDSC" : r'/home/gputwo/bhklab/kevint/cinet/test_data/GDSC_Test_Data/'
}

# Set the location of the hparam tuning data files
deepCINET_hparam_tuned = json.load(open('/home/gputwo/bhklab/kevint/cinet/results/hparam_tuning_delta_DeepCINET.json'))
ECINET_hparam_tuned = json.load(open('/home/gputwo/bhklab/kevint/cinet/results/hparam_tuning_delta_ECINET.json'))

#### GET RESULTS ####
data={}

for param in deepCINET_hparam_tuned:
    drug = param.replace("CCLE_", "")
    # Models
    model_DC = deepCINET(device='gpu', delta=deepCINET_hparam_tuned[param]['best_params']['delta'])
    model_EC = ECINET(device='gpu', delta=ECINET_hparam_tuned[param]['best_params']['delta'])
    # TRAIN the Models
    file_list = os.listdir(train_dir)
    train_file = list(filter(lambda x: drug in x, file_list))[0]
    # Prepare Input Data
    df = pd.read_csv(train_dir + train_file).set_index('cell_line')
    X = df.iloc[:,1:]
    y = df.iloc[:,0]
    model_DC.fit(X,y)
    model_EC.fit(X,y)
    for key in test_directories: 
        test_dir = test_directories[key]
        file_list_2 = os.listdir(test_dir)
        test_file = list(filter(lambda x: drug in x, file_list_2))[0]
        df = pd.read_csv(test_dir + test_file).iloc[:,1:]
        df.values[:] =  StandardScaler().fit_transform(df)
        DC_result = model_DC.score(df.iloc[:, 1:], df.iloc[:, 0]) 
        EC_result = model_EC.score(df.iloc[:, 1:], df.iloc[:, 0]) 
        if drug in data: 
            data[drug][key] = {
                "deepCINET" : DC_result,
                "ECINET" : EC_result
            }
        else: 
            data[drug] = {
                key : {
                    "deepCINET" : DC_result,
                    "ECINET" : EC_result
                }
            }

# Save the results as a json dict

with open('results.json', 'w') as fp:
    json.dump(data, fp)
