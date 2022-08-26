# cinet

Scikit-Learn interface for CINET PyTorch siamese neural network. 

DeepCINET is a deep "siamese" neural network architecture, where a contrastive loss function is used to learn feature weights that maximally discriminate relative response/target between valid pairs of training data. A hyper-parameter, delta, is used to define what a valid pair is by setting a minimum difference in response/target value for pairs to be included in model training, with the intuition that useful weights cannot be learned from samples that are too close together in response-space. 

Concordance index is then used to assess rank accuracy. Concordance index was chosen because it is a non-parametric statistic that does not make 
assumptions on data distributon or homoscedasticity. It can detect non-linear, monotonic associations.

ECINET is a one-dimensional neural network, which makes it essentially a linear regression model with regularization. It is comparable to model architectures like ElasticNet. It can be used
to assess if improved performance is delivered by the added complexity of DeepCINET.

Note, however, that siamese networks go hand-in-hand with few shot learning approaches. The idea is that features learned from large data in CINET can then be applied to learning done 
on smaller real-world data in a transfer learning approach. 

An initial implementation, trained on gene set expression data from cancer cell lines and meant to predict drug sensitivity rank, is available on the BHKLab's public GitHub at https://github.com/bhklab/cinet. 

## Installation

```bash
$ pip3 install cinet
```

## Usage

CINET can be used like any other Scikit-Learn model. 

```python
# Import CINET
from cinet import *

# Create a DeepCINET model
model = deepCINET()
# Or, create an ECINET model
model = ECINET()

# Standard Scikit-Learn syntax
model.fit(X,y)
model.predict(X)
model.score(X,y)

# You can use it with things like GridSearchCV easily
GridSearchCV(deepCINET(device='cpu', batch_size=2**12), param_grid, refit = True, verbose = 3,n_jobs=3)
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`cinet` was created by Kevin Tabatabaei and Christopher Eeles. It is licensed under the terms of the MIT license.

## Credits

`cinet` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
