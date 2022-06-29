# TRAIN TEST
import torch
from cinet import *
import numpy as np

model = cinet('cinet2.ckpt')
model.set_params()
X = getCINETSampleInput()
model.fit(X)

# X = model.gene_data.gene_exprs

for i in range(1,30): 
    test_tensor = torch.tensor(np.array(model.gene_data.gene_exprs[i], dtype=np.float32).reshape(-1,1)).T
    model.predict(test_tensor)

# py_model = model.getPytorchModel()


