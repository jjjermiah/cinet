print("starting")
import torch
from cinet import *
import numpy as np
model = cinet()
model.set_params()
X = getCINETSampleInput()
model.fit(X)
py_model = model.getPytorchModel()
test = np.array(X.iloc[4].values[2:], dtype=np.float16)
