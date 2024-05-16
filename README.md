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

## Data sources

DeepCINET's training datasets are composed of the Cancer Cell Line Encyclopedia (CCLE, https://www.orcestra.ca/pset/10.5281/zenodo.3905461), and the Cancer Therapeutics Response Portal (CTRP-v2, https://www.orcestra.ca/pset/10.5281/zenodo.7826870). On the other hand, the testing datasets include the Genentech Cell Line Screening Initiative (gCSI, https://www.orcestra.ca/pset/10.5281/zenodo.7829857) and the second version of the Genomics of Drug Sensitivity in Cancer (GDSC-v2, https://www.orcestra.ca/pset/10.5281/zenodo.5787145). All PSet R objects were downloaded from Orcestra (https://www.orcestra.ca/). An extra resource used during the execution of the project was the COSMIC Cancer Gene Census (https://cancer.sanger.ac.uk/census), to select genes related to cancer development. The CCLE, gCSI and GDSC-v2 datasets contain both RNA-Seq data as well as drug response (AAC) data, while the CTRP-v2 dataset exclusively contains drug response data.

## Data curation process
The end goal is to generate **one file for each drug**, containing cell-lines' AAC and RNA-Seq values. The curation process executed followed the following steps:

1. Extraction of _csv_ files from PSets using R files. For CCLE, gCSI, and GDSC-v2, the **OrcestraDatasetCuration.R** code extracts RNA-Seq and AAC values in two separate *csv* files. For CTRP-v2, the code **CTRP-curation.R** extracts the experiments, drugs, and cell-lines *csv* files.
2. Obtention of the **common drugs** between the **gCSI** and **GDSC-v2** datasets.
3. Extraction from the CTRP-v2 experiments *csv* file of the list of cell-lines treated with any of the drugs obtained from step 2, as well as their AAC drug response data to that drug.
4. Intersection of **common cell-lines** between the **CCLE** and **CTRP-v2** datasets. For those cell-lines, concatenate RNA-Seq data from CCLE to CTRP-v2's drug response data.
5. Obtention of **common genes** across the three final datasets, **CCLE-CTRP-v2**, **gCSI**, and **GDSC-v2**. Be careful with gene naming! Make sure to eliminate the *version* of the gene, i.e: (~~ENSG00000088038.18~~, ENSG00000088038)
6. Intersection of the genes obtained from step 5 with the ones in the COSMIC Cancer Gene Census.
7. Lastly, generation of **one *csv* file per drug for each dataset**, each containing cell-lines tested with that drug in that study. Each file contains AAC response values of those cell-lines to the drug at hand, as well as RNA-Seq gene expression data for Cancer Gene Census genes. 

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`cinet` was created by Kevin Tabatabaei and Christopher Eeles. It is licensed under the terms of the MIT license.

## Credits

`cinet` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
