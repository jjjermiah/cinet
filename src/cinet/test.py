print("starting")
from cinet import *
model = cinet()
model.set_params()
X = getCINETSampleInput()
model.fit(X)
py_model = model.getPytorchModel()
