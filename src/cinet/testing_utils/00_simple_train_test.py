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
file = file_dir + file_list[0]

# Set the type of model
model = deepCINET(device='gpu')

# Train the model

# Prepare Input Datacd doc
train_df = pd.read_csv(file).set_index('cell_line')
X = train_df.iloc[:,1:]
y = train_df.iloc[:,0]

# Fit the model
model.fit(X,y)

# Set the test file
# test_file = '/home/gputwo/bhklab/kevint/cinet/test_data/gCSI_Test_Data/gene_gCSI_rnaseq_Erlotinib_response.csv'
test_file = '/home/gputwo/bhklab/kevint/cinet/test_data/GDSC_Test_Data/gene_GDSC_rnaseq_Erlotinib_response.csv'

# Test the model

test_df = pd.read_csv(test_file).iloc[:,1:]
test_df.values[:] =  StandardScaler().fit_transform(test_df)
model.score(test_df.iloc[:, 1:], test_df.iloc[:, 0]) 

# Alternately, instead of model.score(X,y) you can use model.predict(X)
# model.predict(df.iloc[:, 1:])

