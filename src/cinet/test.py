# TRAIN TEST
import torch
from cinet import *
import numpy as np
import os
import pandas as pd
import json
import sys
from io import StringIO


model = cinet('cinet2.ckpt')
model.set_params()

file_list = os.listdir(r'/home/gputwo/bhklab/kevint/cinet/data/')

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




# X = model.gene_data.gene_exprs
# for i in range(1,30): 
#     test_tensor = torch.tensor(np.array(model.gene_data.gene_exprs[i], dtype=np.float32).reshape(-1,1)).T
#     print(model.predict(test_tensor))

# py_model = model.getPytorchModel()


