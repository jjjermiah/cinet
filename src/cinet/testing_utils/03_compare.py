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
from sklearn.linear_model import ElasticNet
from lassonet import LassoNetRegressor, LassoNetRegressorCV

# Set the training data directory
file_dir = r'/home/gputwo/bhklab/kevint/cinet/train_data/'
file_list = os.listdir(file_dir)
file = file_dir + file_list[0]

# Set the type of model
modelDC = deepCINET(device='gpu')
modelEC = ECINET(device='gpu')
# Need to modify the 'score' function for the two below... or just simply get their
# predictions and put them into a concordance_index test by myself
modelLN = LassoNetRegressor() # Need to tune lambda_start and lambda
modelEN = ElasticNet() # Need to tune alpha, l1_ratio

# Train the model

# PREPARE INPUT DATA
train_df = pd.read_csv(file).set_index('cell_line')
X = train_df.iloc[:,1:]
# Normalize Gene Expression
X = (X - np.mean(X,axis=0)) / np.std(X,axis=0)
y = train_df.iloc[:,0]

# FIT THE MODEL
# modelDC.fit(X,y)
# modelEC.fit(X,y)
modelLN.fit(X,y)
# modelEN.fit(X,y)
print("done fitting")

# SET UP TEST FILE
# test_file = '/home/gputwo/bhklab/kevint/cinet/test_data/gCSI_Test_Data/gene_gCSI_rnaseq_Erlotinib_response.csv'
test_file = '/home/gputwo/bhklab/kevint/cinet/test_data/GDSC_Test_Data/gene_GDSC_rnaseq_Erlotinib_response.csv'
test_df = pd.read_csv(test_file).iloc[:,1:]
test_df_X = test_df.iloc[:, 1:]
# Normalize Gene Expression
test_df_X = (test_df_X - np.mean(test_df_X,axis=0)) / np.std(test_df_X,axis=0)
test_df_Y = test_df.iloc[:, 0]

# TEST MODEL
# modelDC.score(test_df_X, test_df_Y)
# modelEC.score(test_df_X, test_df_Y)
print("About to score")
print(modelLN.score(test_df_X, test_df_Y))
# modelEN.score(test_df_X, test_df_Y)