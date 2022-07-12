from .models import *

from random import randint
import sklearn
import pandas as pd
import numpy as np
import argparse

## FIXME:: modularize these imports and remove as many as possible!

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.utils.data

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cloud_io import load as pl_load


def getCINETSampleInput(file='./gene_CCLE_rnaseq_Bortezomib_response.csv'): 
    # return pd.read_csv('./gene_CCLE_rnaseq_5-Fluorouracil_response.csv')
    return pd.read_csv(file)

class cinet(sklearn.base.BaseEstimator):
    def __init__(self, modelPath='',
    batch_size=256, num_workers=8, folds=5, use_folds=5, momentum=5, 
    weight_decay=5, sc_milestones=[1,2,5,15,30], sc_gamma=0.35,
    delta=0, dropout=0.4, learning_rate=0.01):
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


    def set_params(self, **params):
        self.batch_size = params['batch_size'] if 'batch_size' in params else 256
        self.num_workers = params['num_workers'] if 'num_workers' in params else 8
        self.folds = params['folds'] if 'folds' in params else 5
        self.use_folds = params['use_folds'] if 'use_folds' in params else False
        self.momentum = params['momentum'] if 'momentum' in params else 0.9
        self.weight_decay = params['weight_decay'] if 'weight_decay' in params else 0
        self.sc_milestones = params['sc_milestones'] if 'sc_milestones' in params else [1,2,5,15,30]
        self.sc_gamma = params['sc_gamma'] if 'sc_gamma' in params else 0.35
        self.delta = params['delta'] if 'delta' in params else 0
        self.dropout = params['dropout'] if 'dropout' in params else 0.4
        self.learning_rate = params['learning_rate'] if 'learning_rate' in params else 0.01

    def fit(self, X=None, y=None):
         # Setup parsers
        self.parser = argparse.ArgumentParser()
        data_arg = self.add_argument_group('Data')

        ## DATA ############
        data_arg.add_argument("--num-workers", default=self.num_workers, type=int)
        data_arg.add_argument("--batch-size", default=self.batch_size, type=int)
        data_arg.add_argument("--folds", default=self.folds, type=int)
        data_arg.add_argument('--use-volume-cache', action='store_true')
        data_arg.add_argument('--accumulate-grad-batches', default=1, type=int)

        ## TRAINING ########
        train_arg = self.add_argument_group('Train')
        train_arg.add_argument("--min-epochs", default=0, type=int)
        train_arg.add_argument("--min-steps", default=None, type=int)
        train_arg.add_argument("--max-epochs", default=12, type=int)  # 12 for tuning
        train_arg.add_argument("--max-steps", default=None, type=int)
        train_arg.add_argument("--check-val-every-n-epoch", default=1, type=int)
        train_arg.add_argument('--auto-find-lr', action='store_true', default=False)
        train_arg.add_argument("--gpus", default=0, type=int)
        ####################

        ## DEBUG ###########
        debug_arg = self.add_argument_group('Debug')
        debug_arg.add_argument("--overfit-pct", default=0, type=float)
        ####################

        ## MISC ############
        misc_arg = self.add_argument_group('Misc')
        misc_arg.add_argument("--seed", default=520, type=int)
        ####################

        self.parser = DeepCINET.add_model_specific_args(self, self.parser)
        self.args, self.unparsed = self.parser.parse_known_args()

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        hdict = vars(self.args)
        self.hparams = argparse.Namespace(**hdict)


        self.config = {
            'hidden_one': 128,
            'hidden_two': 512,
            'hidden_three': 128,
            'hidden_four': 0,  # TODO: Figure out why
            'dropout': self.dropout,
            'lr': self.learning_rate,
            'batchnorm': True,
            # 'dat_size': self.gene_data.gene_num(),
        }



        self.dataSet = X
        self.gene_data = Dataset(self.dataSet, False)
        train_idx, val_idx = train_test_split(list(range(self.gene_data.__len__())), test_size=0.2)

        self.config['dat_size'] = self.gene_data.gene_exprs.shape[1]

        models = []

        # for delta in [0, 0.07491282]:
        train_dl = torch.utils.data.DataLoader(
            Dataset(X, True, self.delta, train_idx),
            batch_size=self.hparams.batch_size, 
            shuffle=True, 
            num_workers=self.hparams.num_workers,
        )

        val_dl = torch.utils.data.DataLoader(
            Dataset(X, True, self.delta, val_idx),
            batch_size=self.hparams.batch_size, 
            shuffle=True, 
            num_workers=self.hparams.num_workers,
        )

        filename_log = f'Vorinostat-delta={self.delta:.3f}'
        checkpoint_callback = ModelCheckpoint(
            monitor='val_ci',
            dirpath='./Saved_models/DeepCINET/rnaseq/',
            filename=filename_log,
            save_top_k=1,
            mode='max'
        )

        self.siamese_model = DeepCINET(hparams=self.hparams, config=self.config)
        trainer = Trainer(min_epochs=self.hparams.min_epochs,
                          max_epochs=self.hparams.max_epochs,
                          min_steps=self.hparams.min_steps,
                          max_steps=self.hparams.max_steps,
                          gpus=1,
                          # NEW LINE I ADDED - KEVIN
                          # to run on my computer
                          accelerator='gpu',
                          accumulate_grad_batches=self.hparams.accumulate_grad_batches,
                          # distributed_backend='dp',
                          weights_summary='full',
                          # enable_benchmark=False,
                          num_sanity_val_steps=0,
                          # auto_find_lr=hparams.auto_find_lr,
                          callbacks=[EarlyStopping(monitor='val_ci', mode="max", patience=5),
                                     checkpoint_callback],
                          check_val_every_n_epoch=self.hparams.check_val_every_n_epoch)
        # overfit_pct=hparams.overfit_pct)

        trainer.fit(self.siamese_model,
                    train_dl,
                    val_dl) 
        if self.modelPath != '': 
            torch.save(self.siamese_model, self.modelPath)


    def predict(self, X):
        if self.modelPath != '': 
            self.siamese_model = torch.load(self.modelPath)
            self.siamese_model.eval()
        return self.siamese_model.fc(X)

    def getPytorchModel(self):
        return self.siamese_model if self.siamese_model is not None else None
    
    def getFC(self):
        return self.fc if self.fc is not None else None


    # HELPER SUB-CLASSES AND SUB-FUNCTIONS
    def verifyType(obj, type):
        if not isinstance(obj, type):
            # Do something, throw an error
            pass

    def add_argument_group(self, name):
        arg = self.parser.add_argument_group(name)
        self.arg_lists.append(arg)
        return arg

   
