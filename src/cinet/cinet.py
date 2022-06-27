from random import randint
import sklearn
import pandas as pd
import numpy as np

## FIXME:: modularize these imports and remoave as many as possible!

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from lifelines.utils import concordance_index
from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.utils.data

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
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
        for i in range(len(layers_size)-1):
            if i == 0:
                curr_dropout = 0
            else:
                curr_dropout = dropout
            if batchnorm:
                layer1 = nn.Sequential(
                    nn.Linear(layers_size[i], layers_size[i+1]),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(layers_size[i+1]),
                    nn.Dropout(curr_dropout)
                )
            else:
                layer1 = nn.Sequential(
                    nn.Linear(layers_size[i], layers_size[i+1]),
                    nn.LeakyReLU(),
                    # Residual(layers_size[i], layers_size[i+1], last_layer = (i+1 == len(layers_size) - 1)),
                    nn.Dropout(curr_dropout)
                )
            self.layers.append(layer1)

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


class DeepCINET(pl.LightningModule):
    """ Base class for our DeepCINET implemented in pytorch lightning
    Provides methods to train and validate as well as configuring the optimizer
    scheduler.
    """

    def __init__(self, hparams, config, data_dir=None, linear=False):
        super(DeepCINET, self).__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)
        # to be tuned hyper-parameters
        self.data_dir = data_dir or os.getcwd()
        self.hidden_one = config["hidden_one"]
        self.hidden_two = config["hidden_two"]
        self.hidden_three = config["hidden_three"]
        self.hidden_four = config["hidden_four"]
        self.data_sz = config["dat_size"]
        if linear:
          self.ratio = config["ratio"]
          self.reg_contr = config["reg_contr"]
        self.layers_size = [i for i in [self.data_sz, self.hidden_one, self.hidden_two, self.hidden_three, self.hidden_four, 1] if i != 0]
        self.dropout = config["dropout"]
        self.lr = config["lr"]
        self.batchnorm = config["batchnorm"]

        self.t_steps = 0
        self.cvdata = []
        self.best_val_loss = 0
        self.best_val_ci = -1 # max 1
        self.test_results = {}
        self.criterion = nn.MarginRankingLoss()
        self.convolution = nn.Identity()
        self.linear = linear
        if self.linear:
          self.fc = FullyConnectedLinear(self.layers_size, self.dropout, self.batchnorm)
        else:
          self.fc = FullyConnected(self.layers_size, self.dropout, self.batchnorm)
        self.log_model_parameters()

    def forward(self, geneA, geneB):
        tA = self.fc(geneA)
        tB = self.fc(geneB)
        z = (tA - tB)
        # return torch.sigmoid(z)
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
          reg = (self.ratio * torch.abs(torch.cat(weights)).sum()) + ((1-self.ratio) * torch.square(torch.cat(weights)).sum())
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
        self.log_dict(tensorboard_logs, prog_bar = True)
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
          reg = (self.ratio * torch.abs(torch.cat(weights)).sum()) + ((1-self.ratio) * torch.square(torch.cat(weights)).sum())
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

        # TODO: This does not work, as lightning does not update the
        #drug_response = np.concatenate([x['Drug_response'] for x in outputs])
        #drug_response_pred = np.concatenate([x['Drug_response_pred'] for x in outputs])
        ## Have samples been averaged out??
        # print("", file=sys.stderr)
        # print("Total size", file=sys.stderr)
        # print(events.shape, file=sys.stderr)
        # print("", file=sys.stderr)
        # print(energies, file=sys.stderr)

        # ci = concordance_index(drug_response, drug_response_pred)
        # print(ci)
        # tensorboard_logs = {'val_CI': ci}
        
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
        self.log('best_loss', self.best_val_loss, prog_bar = False)
        self.log('best_val_ci', self.best_val_ci, prog_bar = False)
        self.log('val_loss', val_avg_loss, prog_bar = True)
        self.log('val_ci', ci, prog_bar = True)


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
                                    lr=self.lr)#,
                                    #momentum=self.hparams.momentum,
                                    #weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.hparams.sc_milestones,
            gamma=self.hparams.sc_gamma)

        return [optimizer], [scheduler]

    def log_model_parameters(self):
        print("PARAMETERS**********************************************")
        print("Convolution layer parameters: %d" % (count_parameters(self.convolution)))
        print("FC layer parameters: %d" % (count_parameters(self.fc)))
        print("********************************************************")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
