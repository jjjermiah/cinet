from .models import *
from scipy import stats

from random import randint
import sklearn
import pandas as pd
import numpy as np
import argparse

from abc import ABCMeta, abstractmethod, abstractstaticmethod

from lifelines.utils import concordance_index

## FIXME:: modularize these imports and remove as many as possible!

from sklearn.model_selection import train_test_split

import torch
import torch.utils.data

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

def abstractattr(f):
    return property(abstractmethod(f))

# TODO: Figure out what the abc stuff is up to
# TODO: Make it so that there is a parameter (tuple) called hidden_dims 
# TODO: Reference LassoNet to see other parameters I could take in
# TODO; Make documentation 
# TODO: Figure out where type validation is done. There is no "set_params" in lassoNET
class BaseCINET(sklearn.base.BaseEstimator, metaclass=ABCMeta):
    def __init__(self,
    *, 
    modelPath='',
    batch_size=256,
    num_workers=8,
    folds=5, 
    use_folds=False, 
    momentum=5.0, 
    weight_decay=5.0, 
    sc_milestones=[1,2,5,15,30], 
    sc_gamma=0.35,
    delta=0.0, 
    dropout=0.4, 
    learning_rate=0.01, 
    device='cpu',
    seed=420):
        self.arg_lists = []
        self._estimator_type = 'classifier'
        self.modelPath = modelPath
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.folds = folds
        self.use_folds = use_folds
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.sc_milestones = sc_milestones
        self.sc_gamma = sc_gamma
        self.delta = delta
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = device
        self.seed = seed


    def _validate_params(self): 
        assert isinstance(self.batch_size, int), 'batch_size must be of type int'
        assert isinstance(self.num_workers, int), 'num_workers must be of type int'
        assert isinstance(self.folds, int), 'folds must be of type int'
        assert isinstance(self.use_folds, bool), 'use_folds must be of type bool'
        assert isinstance(self.momentum, float), 'momentum must be of type float'
        assert isinstance(self.weight_decay, float), 'weight_decay must be of type float'
        assert isinstance(self.sc_milestones, list), 'sc_milestones must be of type list'
        assert isinstance(self.sc_gamma, float), 'sc_gamma must be of type float'
        assert isinstance(self.delta, float), 'delta must be of type float'
        assert isinstance(self.dropout, float), 'dropout must be of type float'
        assert isinstance(self.learning_rate, float), 'learning_rate must be of type float'
        assert isinstance(self.device, str), 'device must be of type str'
        assert (self.device in ['cpu', 'gpu']), 'device must be either "cpu" or "gpu"'
        assert isinstance(self.seed, int), 'seed must be of type int'


    def fit(self, X=None, y=None): 
        self._validate_params()
        print("🚀🚀🚀🚀TESTING WITH HYPERPARAMETERS🚀🚀🚀🚀")
        print("delta", self.delta)

        self.hyperparams = {
            "num_workers": self.num_workers, 
            "batch_size" : self.batch_size, 
            "folds" : self.folds, 
            "accumulate_grad_batches": 1, 
            "min_epochs": 0, 
            "min_steps" : None,
            "max_epochs" : 12, 
            "max_steps" : None, 
            "check_val_every_n_epoch" : 1, 
            "gpus" : 0,
            "overfit_pct" : 0,
            "seed" : self.seed,
            "sc_milestones" : self.sc_milestones,
            "sc_gamma" : self.sc_gamma,
        }

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        np.random.seed(self.hyperparams["seed"])
        torch.manual_seed(self.hyperparams["seed"])

        self.config = self.getConfig()

        # Check if both data have same # of rows 
        if len(X) != len(y):
            raise Exception("X and y values are not of the same length")
        

        combined_df = pd.concat([X,y],axis=1)
        combined_df.columns.values[-1] = 'target'

        print(combined_df.shape, combined_df.columns)

        # Check if the combined dataframe is the right size
        if len(combined_df) != len(X): 
            raise Exception("X and y values must have the same indices")

        self.dataSet = combined_df
        self.gene_data = Dataset(self.dataSet, False, self.batch_size)
        train_idx, val_idx = train_test_split(list(range(self.gene_data.__len__())), test_size=0.2)

        self.config['dat_size'] = self.gene_data.gene_exprs.shape[1]

        train_dl = torch.utils.data.DataLoader(
            Dataset(combined_df, True, self.batch_size, self.delta, train_idx),
            batch_size=self.hyperparams['batch_size'], 
            shuffle=True, 
            num_workers=self.hyperparams['num_workers'],
            multiprocessing_context='fork',
        )

        val_dl = torch.utils.data.DataLoader(
            Dataset(combined_df, True, self.batch_size, self.delta, val_idx),
            batch_size=self.hyperparams['batch_size'], 
            shuffle=True, 
            num_workers=self.hyperparams['num_workers'],
            multiprocessing_context='fork',
        )

        # TODO: Remove this? Hard-coded stuff here. 
        # filename_log = f'Vorinostat-delta={self.delta:.3f}'
        # checkpoint_callback = ModelCheckpoint(
        #     monitor='val_ci',
        #     dirpath='./Saved_models/DeepCINET/rnaseq/',
        #     filename=filename_log,
        #     save_top_k=1,
        #     mode='max'
        # )

        self.siamese_model = DeepCINET(hyperparams=self.hyperparams, config=self.config, linear=self.isLinear())
        trainer = Trainer(min_epochs=self.hyperparams['min_epochs'],
                        max_epochs=self.hyperparams['max_epochs'],
                        min_steps=self.hyperparams['min_steps'],
                        max_steps=self.hyperparams['max_steps'],
                        gpus=1,
                        devices=1,
                        accelerator=self.device,
                        accumulate_grad_batches=self.hyperparams['accumulate_grad_batches'],
                        # distributed_backend='dp',
                        weights_summary='full',
                        # enable_benchmark=False,
                        num_sanity_val_steps=0,
                        # auto_find_lr=hparams.auto_find_lr,
                        #   callbacks=[EarlyStopping(monitor='val_ci', mode="max", patience=5),
                        #              checkpoint_callback],
                        check_val_every_n_epoch=self.hyperparams['check_val_every_n_epoch'])
        # overfit_pct=hparams.overfit_pct)

        trainer.fit(self.siamese_model,
                    train_dl,
                    val_dl) 
        if self.modelPath != '': 
            torch.save(self.siamese_model, self.modelPath)
    
    def predict(self, X):
        if type(X) == pd.DataFrame: 
            X = torch.FloatTensor(X.values)
        if self.modelPath != '': 
            self.siamese_model = torch.load(self.modelPath)
            self.siamese_model.eval()
        return self.siamese_model.fc(X)

    def score(self, X=None, y=None):
        if type(X) == pd.DataFrame: 
            X = torch.FloatTensor(X.values)
        temp_list = self.predict(X).detach().numpy().tolist()
        final_list = []
        for t in temp_list: 
            final_list.append(t[0])

        # return stats.spearmanr(y,final_list)
        return concordance_index(y,final_list)

    # HELPER SUB-CLASSES AND SUB-FUNCTIONS

    def add_argument_group(self, name):
        arg = self.parser.add_argument_group(name)
        self.arg_lists.append(arg)
        return arg
    
    # DEBUG TOOLS 
    
    def getPytorchModel(self):
        return self.siamese_model if self.siamese_model is not None else None


    ### TO BE IMPLEMENTED BY INHERITING CLASSES ###

    @abstractattr
    def getConfig(self): 
        """return a configuration object for the neural network"""
        raise NotImplementedError
    
    @abstractattr
    def isLinear(self): 
        """return a boolean value indicating whether the model is linear, and should use 
        FullyConnectedLinear as opposed to FullyConnected"""
        raise NotImplementedError


class deepCINET(BaseCINET): 
    def getConfig(self): 
        return  {
            'hidden_one': 128,
            'hidden_two': 512,
            'hidden_three': 128,
            'hidden_four': 0,  
            'dropout': self.dropout,
            'lr': self.learning_rate,
            'batchnorm': True,
        }
    
    def isLinear(self): 
        return False

class ECINET(BaseCINET): 
    def getConfig(self): 
        return {
            'hidden_one': 0,
            'hidden_two': 0,
            'hidden_three': 0,
            'hidden_four': 0, 
            'dropout': self.dropout,
            'lr': self.learning_rate,
            'batchnorm': False,
            # TODO: Not hardcode these two following values
            'ratio': 0.4, 
            'reg_contr': 0.4,
        }

    def isLinear(self): 
        return True


