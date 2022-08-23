from random import randint
import sklearn
import pandas as pd
import numpy as np
import os
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

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining, HyperBandForBOHB
from ray.tune.integration.pytorch_lightning import TuneCallback


class FullyConnected(nn.Module):
    """
    Fully connected network architecture for CINET models. This corresponds
    to the DeepCINET method.
    """
    def __init__(self, layers_size, dropout, batchnorm):
        super(FullyConnected, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_size) - 1):
            if i == 0:
                curr_dropout = 0
            else:
                curr_dropout = dropout
            
            # define block with FC layer
            block = [nn.Linear(layers_size[i], layers_size[i + 1])]
            
            # activation layer
            if i == len(layers_size) - 2: #last layer
                block.append(nn.Sigmoid())
            else:
                block.append(nn.LeakyReLU())
                # batchnorm layer 
                if batchnorm:
                    block.append(nn.BatchNorm1d(layers_size[i + 1]))
                # dropout layer
                block.append(nn.Dropout(curr_dropout))

            self.layers.append(nn.Sequential(*block))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x

class FullyConnectedLinear(nn.Module):
    def __init__(self, layers_size, dropout, batchnorm):
        super(FullyConnectedLinear, self).__init__()
        self.layers = nn.Sequential(
        nn.Linear(layers_size[0], layers_size[1])
      )


    def forward(self, x):
      '''Forward pass'''
      x = x.view(x.size(0), -1)
      for layer in self.layers:
          x = layer(x)
      return x

class Dataset(torch.utils.data.Dataset):
    """Data set class which returns a pytorch data set object
        Returns a iterable data set object extending from the pytorch dataset
        object.
    """

    def __init__(self, dataframe, is_train, batch_size, delta=0, idxs=None):
        self.gene_exprs = dataframe
        self.batch_size = batch_size
        if idxs is not None:
            self.gene_exprs = self.gene_exprs.iloc[idxs]
        self.drug_resps = self.gene_exprs["target"].to_numpy()
        # self.cell_lines = self.gene_exprs["cell_line"].to_numpy()
        self.cell_lines = self.gene_exprs.index.values.tolist()
        # self.gene_exprs = self.gene_exprs.drop(["target", "cell_line"], axis=1).to_numpy()
        self.gene_exprs = self.gene_exprs.drop(["target"], axis=1).to_numpy()
        self.gene_exprs = (self.gene_exprs - np.mean(self.gene_exprs, axis=0)) / np.std(self.gene_exprs, axis=0)


        print("SHAPE2: ", self.gene_exprs.shape)

        self._is_train = is_train
        self.delta = delta
        self._sample_list = self._build_pairs(self.delta)

    def __len__(self):
        return len(self._sample_list)

    def gene_num(self):
        return len(self.gene_exprs[0])

    def __getitem__(self, index):
        return self.train_item(index) if self._is_train else self.test_item(index)

    def train_item(self, pair_idx):
        row = self._sample_list[pair_idx]
        gene1 = self._load_item(row['idxA'])
        gene2 = self._load_item(row['idxB'])
        label = torch.tensor(row['label'], dtype=torch.float32)
        return {'geneA': gene1,
                'geneB': gene2,
                'labels': label}

    def test_item(self, idx):
        gene = self._load_item(idx)
        response = self._load_response(idx)
        # cell_line = self._load_cell_line(idx)
        return {'gene': gene,
                'response': response,
                'cell_line': idx}

    def _load_item(self, idx):
        """ Function to load the features of a cell line
        :param idx: the cell line index in our input csv
        :return: returns a gene expression variable
        """
        gene = self.gene_exprs[idx]
        gene = torch.tensor(gene.copy(), dtype=torch.float32)
        return gene

    def _load_response(self, idx):
        response = self.drug_resps[idx]
        response = torch.tensor(response.copy(), dtype=torch.float32)
        return response

    # def _load_cell_line(self, idx):
    #     cell_lines_selected = self.cell_lines[idx]
    #     return cell_lines_selected

    def _build_pairs(self, delta):
        ''' build pairs of indices and labels for training data
        '''
        if self._is_train:
            return self.get_concordant_pair_list(delta)
        else:
            return self.drug_resps

    def get_concordant_pair_list(self, delta):
        pairs = []
        size = self.gene_exprs.shape[0]
        print("SIZE: ", size)
        for i in range(size - 1):
            for j in range(i + 1, size, 1):
                if (abs(self.drug_resps[i] - self.drug_resps[j]) > delta): 
                    pairs.append({'idxA': i, 'idxB': j,
                                    'label': self.get_relationship_from_index(i, j)})
        # Quick and dirty fix
        # Duplicate the very last row if there's only one row to be fed into a batch
        # i.e. total length / batch size leads to a remainder of one
        # This is required for batchnorm to work
        # (Batchnorm can't work on just one row)
        if len(pairs) % self.batch_size == 1:
            print("🏙 Adding one!!")
            pairs.append(pairs[-1])

        return pairs

    def get_relationship_from_index(self, i, j):
        '''
        check if drug reponse at index i is greater than drug response at index j
        '''
        drug_i = self.drug_resps[i]
        drug_j = self.drug_resps[j]
        return int(drug_i > drug_j)

class DeepCINET(pl.LightningModule):
    """ Base class for our DeepCINET implemented in pytorch lightning
    Provides methods to train and validate as well as configuring the optimizer
    scheduler.
    """

    def __init__(self, hyperparams, config, data_dir=None, linear=False):
        super(DeepCINET, self).__init__()
        self.hyperparams = hyperparams
        # self.save_hyperparameters(hparams)

        # to be tuned hyper-parameters
        self.data_dir = data_dir or os.getcwd()
        self.nnHiddenLayers = config["nnHiddenLayers"]
        self.data_sz = config["dat_size"]
        if linear:
            self.ratio = config["ratio"]
            self.reg_contr = config["reg_contr"]
        print(self.nnHiddenLayers)
        self.layers_size = [i for i in
                            [self.data_sz, *list(self.nnHiddenLayers), 1] if
                            i != 0]
        self.dropout = config["dropout"]
        self.lr = config["lr"]
        self.batchnorm = config["batchnorm"]

        self.t_steps = 0
        self.cvdata = []
        self.best_val_loss = 0
        self.best_val_ci = -1  # max 1
        self.test_results = {}
        self.criterion = nn.MarginRankingLoss()
        self.convolution = nn.Identity()
        self.linear = linear

        if self.linear:
            self.fc = FullyConnectedLinear(self.layers_size, self.dropout, self.batchnorm)
            pass
        else:
            self.fc = FullyConnected(self.layers_size, self.dropout, self.batchnorm)
        self.log_model_parameters()


    def forward(self, geneA, geneB):
        tA = self.fc(geneA)
        tB = self.fc(geneB)
        z = (tA - tB)
        return z

    def training_step(self, batch, batch_idx):
        geneA = batch['geneA']
        geneB = batch['geneB']
        labels = batch['labels']

        output = self.forward(geneA, geneB)
        # labels_hinge = labels.view(-1).detach()
        labels_hinge = torch.where(labels == 0, torch.tensor(-1).type_as(labels), torch.tensor(1).type_as(labels))
        loss = self.criterion(output.view(-1), torch.zeros(labels_hinge.size()).type_as(labels), labels_hinge)

        # Compute L1 and L2 loss component if using ECINET
        if self.linear:
            weights = []
            for parameter in self.parameters():
                weights.append(parameter.view(-1))
            reg = (self.ratio * torch.abs(torch.cat(weights)).sum()) + (
                        (1 - self.ratio) * torch.square(torch.cat(weights)).sum())
            loss += reg * self.reg_contr

        # loggin number of steps
        self.t_steps += 1

        np_output = torch.sigmoid(output.view(-1)).detach()
        output_class = torch.where(np_output < 0.5,
                                   torch.tensor(0).type_as(np_output),
                                   torch.tensor(1).type_as(np_output))
        correct = torch.sum(output_class == labels).type_as(np_output)
        total = torch.tensor(np_output.size(0)).type_as(np_output)
        CI = correct / total

        tensorboard_logs = {'train_loss': loss, 'CI': CI}
        return {'loss': loss, 'custom_logs': tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['custom_logs']['train_loss'].mean() for x in outputs]).mean()
        CI = torch.stack([x['custom_logs']['CI'].mean() for x in outputs]).mean()

        # TODO: This does not work, as lightning does not update the
        # progress bar on training epoch end
        tensorboard_logs = {
            'avg_loss': avg_loss,
            'train_CI': CI}
        self.log_dict(tensorboard_logs, prog_bar=True)
        # return {'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        geneA = batch['geneA']
        geneB = batch['geneB']
        labels = batch['labels']

        output = self.forward(geneA, geneB)
        # labels_hinge = labels.view(-1).detach()
        labels_hinge = torch.where(labels == 0, torch.tensor(-1).type_as(labels), torch.tensor(1).type_as(labels))
        loss = self.criterion(output.view(-1), torch.zeros(labels_hinge.size()).type_as(labels), labels_hinge)

        # Compute L1 and L2 loss component
        if self.linear:
            weights = []
            for parameter in self.parameters():
                weights.append(parameter.view(-1))
            reg = (self.ratio * torch.abs(torch.cat(weights)).sum()) + (
                        (1 - self.ratio) * torch.square(torch.cat(weights)).sum())
            loss += reg * self.reg_contr

        np_output = torch.sigmoid(output.view(-1)).detach()
        output_class = torch.where(np_output < 0.5,
                                   torch.tensor(0).type_as(np_output),
                                   torch.tensor(1).type_as(np_output))
        correct = torch.sum(output_class == labels).type_as(np_output)
        total = torch.tensor(np_output.size(0)).type_as(np_output)
        CI = correct / total

        val_logs = {'val_loss': loss, 'val_CI': CI}

        # TODO: Pytorch currently doesn't reduce the output in validation when
        #       we use more than one GPU, becareful this might not be supported
        #       future versions
        return val_logs

    def validation_epoch_end(self, outputs):
        val_avg_loss = torch.stack([x['val_loss'].mean() for x in outputs]).mean()
        ci = torch.stack([x['val_CI'].mean() for x in outputs]).mean().cpu()

        self.cvdata.append({
            'CI': ci,
            't_steps': self.t_steps
        })

        if self.best_val_ci == -1:
            self.best_val_loss = val_avg_loss
            self.best_val_ci = ci
        else:
            if self.best_val_ci <= ci:
                self.best_val_loss = val_avg_loss
                self.best_val_ci = ci
        self.log('best_loss', self.best_val_loss, prog_bar=False)
        self.log('best_val_ci', self.best_val_ci, prog_bar=False)
        self.log('val_loss', val_avg_loss, prog_bar=True)
        self.log('val_ci', ci, prog_bar=True)

    def test_step(self, batch, batch_idx):
        gene = batch['gene']
        y_true = np.array(batch['response'])
        cell_line = np.array(batch['cell_line'])
        drug_pred = self.fc(gene)

        test_ret_batch = {'cell_line': cell_line, 'y_true': y_true, 'y_hat': drug_pred.numpy()}
        return test_ret_batch

    def test_epoch_end(self, outputs):
        self.test_results["cell_line"] = np.concatenate([x['cell_line'] for x in outputs])
        self.test_results["y_true"] = np.concatenate([x['y_true'] for x in outputs])
        self.test_results["y_hat"] = np.concatenate([x['y_hat'].reshape(-1) for x in outputs])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr)  # ,
        # momentum=self.hparams.momentum,
        # weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.hyperparams['sc_milestones'],
            gamma=self.hyperparams['sc_gamma'])

        return [optimizer], [scheduler]

    def log_model_parameters(self):
        print("PARAMETERS**********************************************")
        print("Convolution layer parameters: %d" % (self.count_parameters(self.convolution)))
        print("FC layer parameters: %d" % (self.count_parameters(self.fc)))
        print("********************************************************")

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)