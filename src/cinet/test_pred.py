

# PREDICT TEST
import torch
from cinet import *
import numpy as np

model = cinet('/home/gputwo/bhklab/kevint/cinet/cinet1.ckpt')
model.set_params()

X = getCINETSampleInput()

for i in range(1,30): 
    test_tensor = torch.tensor(np.array(X.iloc[i].values[2:], dtype=np.float32).reshape(-1,1)).T
    model.predict(test_tensor)


