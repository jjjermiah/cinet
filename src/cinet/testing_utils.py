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

file_list = os.listdir(r'/home/gputwo/bhklab/kevint/cinet/train_data/')
model = ECINET(device='gpu')

data = {}

for file in file_list: 
    name = file.replace('_response.csv','').replace('rnaseq_','').replace('gene_', '')
    df = pd.read_csv('/home/gputwo/bhklab/kevint/cinet/train_data/' + file).set_index('cell_line')
    X = df.iloc[:,1:]
    y = df.iloc[:,0]
    param_grid = { "delta" : [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2] }
    grid = GridSearchCV(ECINET(modelPath= (name + '.ckpt'), device='cpu', batch_size=2**12), param_grid, refit = True, verbose = 3,n_jobs=1)
    grid.fit(X,y)
    data[name] = {
        "best_params" : grid.best_params_,
        "cv_results" : grid.cv_results_,
    }



for key in data:
    for result in data[key]:
        for x in data[key][result]:
            dataType = type(data[key][result][x])
            if dataType == numpy.ndarray: 
                data[key][result][x] = data[key][result][x].tolist()
                



with open('hparam_tuning_delta2.json', 'w') as fp:
    json.dump(data, fp)
   


### PREPARE INPUT DATA
df = pd.read_csv('/home/gputwo/bhklab/kevint/cinet/train_data/' + file_list[1]).set_index('cell_line')
X = df.iloc[:,1:]
y = df.iloc[:,0]

### FIT THE MODEL 
model.fit(X,y)

### TEST THE MODEL 
# df = pd.read_csv('/home/gputwo/bhklab/kevint/cinet/gene_gCSI_rnaseq_Erlotinib_response.csv').iloc[:,1:]
df = pd.read_csv('/home/gputwo/bhklab/kevint/cinet/gene_gCSI_rnaseq_AZD7762_response.csv').iloc[:,1:]
df.values[:] =  StandardScaler().fit_transform(df)
model.score(df.iloc[:, 1:], df.iloc[:, 0]) 


### CV ###
param_grid = { "delta" : [0.0,0.025,0.05,0.075,0.1, ]}
grid = GridSearchCV(deepCINET(modelPath='cinet2.ckpt', device='cpu', batch_size=2**12), param_grid, refit = True, verbose = 3,n_jobs=1)
grid.fit(X,y)
#######








### GET RESULTS 
data={}
for test_directory in [r'/home/gputwo/bhklab/kevint/cinet/train_data/', r'/home/gputwo/bhklab/kevint/cinet/test_data/gCSI_Test_Data/', r'/home/gputwo/bhklab/kevint/cinet/test_data/GDSC_Test_Data/']:
    print('\n' + test_directory + '\n')
    file_list2 = os.listdir(test_directory)
    model = deepCINET(modelPath='checkpoint.ckpt', device='gpu')
    for file in file_list2: 
        name = file.replace('_response.csv','').replace('rnaseq_','').replace('gene_', '')
        name_clean = name.replace('CCLE_','').replace('gCSI_','').replace('GDSC_','')
        model_path = '/home/gputwo/bhklab/kevint/cinet/models/' + 'CCLE_' + name_clean + '.ckpt'
        test_df = pd.read_csv(test_directory + file).set_index('cell_line')
        X = test_df.iloc[:,1:]
        X.values[:] = StandardScaler().fit_transform(X)
        y = test_df.iloc[:,0]
        model = deepCINET(modelPath=model_path, device='gpu')
        print(name)
        value = model.score(X,y)
        print(value)
        if name_clean in data: 
            data[name_clean].append(value)
        else: 
            data[name_clean] = [value]


with open('results.json', 'w') as fp:
    json.dump(data, fp)












data = dict()

for file in file_list:
    print("🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵")
    print(file) 
    data[file] = {
        "success" : False,
        "output_dict" : []
    }

    try: 
        X = pd.read_csv('/home/gputwo/bhklab/kevint/cinet/data/' + file)
        

        model.fit(X)

        temp_list = model.predict(torch.FloatTensor(model.gene_data.gene_exprs)).detach().numpy().tolist()
        temp_list = model.predict(torch.FloatTensor(test2.iloc[:,2:].T)).detach().numpy().tolist()

        final_list = []
        for t in temp_list: 
            final_list.append(t[0])

        data[file] = {
            "success" : True, 
            "output_dict" : final_list
        }
    except: 
        print("❌❌❌❌❌❌Something went wrong❌❌❌❌❌❌❌")

with open('data.json', 'w') as fp:
    json.dump(data, fp)


#####
from scipy import stats

test_cv = pd.read_csv('/home/gputwo/bhklab/kevint/cinet/gene_gCSI_rnaseq_Erlotinib_response.csv')
temp_list = model.predict(torch.FloatTensor(test_cv.iloc[:,2:].values)).detach().numpy().tolist()
final_list = []
for t in temp_list: 
    final_list.append(t[0])

c2 = final_list

c1 = np.asarray(test_cv.iloc[:,1]) 

stats.spearmanr(c1,c2)
concordance_index(c1,c2)
###


# X = model.gene_data.gene_exprs
# for i in range(1,30): 
#     test_tensor = torch.tensor(np.array(model.gene_data.gene_exprs[i], dtype=np.float32).reshape(-1,1)).T
#     print(model.predict(test_tensor))

# py_model = model.getPytorchModel()


