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


model = cinet(modelPath='cinet2.ckpt', device='gpu')

file_list = os.listdir(r'/home/gputwo/bhklab/kevint/cinet/data/')

### PREPARE INPUT DATA
df = pd.read_csv('/home/gputwo/bhklab/kevint/cinet/data/' + file_list[0]).set_index('cell_line')
X = df.iloc[:,2:]
y = df.iloc[:,1]

### FIT THE MODEL 
model.fit(X,y)

### TEST THE MODEL 
df = pd.read_csv('/home/gputwo/bhklab/kevint/cinet/gene_gCSI_rnaseq_Erlotinib_response.csv').iloc[:,1:]
df.values[:] =  StandardScaler().fit_transform(df)
model.score(df.iloc[:,1:], df.iloc[:,0])





#######


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


