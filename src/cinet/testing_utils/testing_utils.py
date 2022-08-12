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


