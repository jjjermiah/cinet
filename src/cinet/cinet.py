from random import randint
import sklearn
import pandas as pd
import numpy as np
import os
import argparse

## FIXME:: modularize these imports and remoave as many as possible!

from sklearn.model_selection import train_test_split

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

def getCINETSampleInput(): 
    return pd.read_csv('./gene_CCLE_rnaseq_5-Fluorouracil_response.csv')

class cinet(sklearn.base.BaseEstimator):
    def __init__(self):
        # super.__init__() # Is this necessary?
        self.arg_lists = []
        self._estimator_type = 'classifier'

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
            'dropout': 0.029255543838070382,
            'lr': 0.03846885084652688,
            'batchnorm': True,
            # 'dat_size': self.gene_data.gene_num(),
        }

    def get_params(self, deep=True):
        return None
        # TODO: Figure this out

    def fit(self, X=None, y=None):
        self.dataSet = X
        self.gene_data = cinet.Dataset(self.dataSet, False)
        train_idx, val_idx = train_test_split(list(range(self.gene_data.__len__())), test_size=0.2)

        self.config['dat_size'] = self.gene_data.gene_exprs.shape[1]

        models = []

        # for delta in [0, 0.07491282]:
        train_dl = self.Create_Dataloader(
            self.Dataset(X, True, self.delta, train_idx),
            self.hparams,
            True)
        val_dl = self.Create_Dataloader(
            self.Dataset(X, True, self.delta, val_idx),
            self.hparams, True)

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
                          accelerator='cpu',
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

    # def predict():
    #     pass

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

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    class FullyConnected(nn.Module):
        """
        Fully connected network architecture for CINET models. This corresponds
        to the DeepCINET method.
        """
        def __init__(self, layers_size, dropout, batchnorm):
            super(cinet.FullyConnected, self).__init__()
            self.layers = nn.ModuleList()
            for i in range(len(layers_size) - 1):
                if i == 0:
                    curr_dropout = 0
                else:
                    curr_dropout = dropout
                if batchnorm:
                    layer1 = nn.Sequential(
                        nn.Linear(layers_size[i], layers_size[i + 1]),
                        nn.LeakyReLU(),
                        nn.BatchNorm1d(layers_size[i + 1]),
                        nn.Dropout(curr_dropout)
                    )
                else:
                    layer1 = nn.Sequential(
                        nn.Linear(layers_size[i], layers_size[i + 1]),
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

    class Dataset(torch.utils.data.Dataset):
        """Data set class which returns a pytorch data set object
            Returns a iterable data set object extending from the pytorch dataset
            object.
        """

        def __init__(self, dataframe, is_train, delta=0, idxs=None):
            self.gene_exprs = dataframe
            if idxs is not None:
                self.gene_exprs = self.gene_exprs.iloc[idxs]
            self.drug_resps = self.gene_exprs["target"].to_numpy()
            self.cell_lines = self.gene_exprs["cell_line"].to_numpy()
            self.gene_exprs = self.gene_exprs.drop(["target", "cell_line"], axis=1).to_numpy()
            self.gene_exprs = (self.gene_exprs - np.mean(self.gene_exprs, axis=0)) / np.std(self.gene_exprs, axis=0)

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
            cell_line = self._load_cell_line(idx)
            return {'gene': gene,
                    'response': response,
                    'cell_line': cell_line}

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

        def _load_cell_line(self, idx):
            cell_lines_selected = self.cell_lines[idx]
            return cell_lines_selected

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
            for i in range(size - 1):
                for j in range(i + 1, size, 1):
                    if (abs(self.drug_resps[i] - self.drug_resps[j]) > delta):
                        pairs.append({'idxA': i, 'idxB': j,
                                      'label': self.get_relationship_from_index(i, j)})
            return pairs

        def get_relationship_from_index(self, i, j):
            '''
            check if drug reponse at index i is greater than drug response at index j
            '''
            drug_i = self.drug_resps[i]
            drug_j = self.drug_resps[j]
            return int(drug_i > drug_j)

    # Run this cell when performing DeepCINET hyperparameter tuning
    class TuneReportCallback(TuneCallback):
        """PyTorch Lightning to Ray Tune reporting callback

        Reports metrics to Ray Tune.

        Args:
            metrics (str|list|dict): Metrics to report to Tune. If this is a list,
                each item describes the metric key reported to PyTorch Lightning,
                and it will reported under the same name to Tune. If this is a
                dict, each key will be the name reported to Tune and the respective
                value will be the metric key reported to PyTorch Lightning.
            on (str|list): When to trigger checkpoint creations. Must be one of
                the PyTorch Lightning event hooks (less the ``on_``), e.g.
                "batch_start", or "train_end". Defaults to "validation_end".

        Example:

        .. code-block:: python

            import pytorch_lightning as pl
            from ray.tune.integration.pytorch_lightning import TuneReportCallback

            # Report loss and accuracy to Tune after each validation epoch:
            trainer = pl.Trainer(callbacks=[TuneReportCallback(
                    ["val_loss", "val_acc"], on="validation_end")])

            # Same as above, but report as `loss` and `mean_accuracy`:
            trainer = pl.Trainer(callbacks=[TuneReportCallback(
                    {"loss": "val_loss", "mean_accuracy": "val_acc"},
                    on="validation_end")])

        """

        def __init__(self,
                     metrics=None,
                     on="validation_end"):
            super(cinet.TuneReportCallback, self).__init__(on)
            if isinstance(metrics, str):
                metrics = [metrics]
            self._metrics = metrics

        def _get_report_dict(self, trainer, pl_module):
            # Don't report if just doing initial validation sanity checks.
            # if trainer.running_sanity_check:
            #    return
            if not self._metrics:
                report_dict = {
                    k: v.item()
                    for k, v in trainer.callback_metrics.items()
                }
            else:
                report_dict = {}
                for key in self._metrics:
                    if isinstance(self._metrics, dict):
                        metric = self._metrics[key]
                    else:
                        metric = key
                    if metric in trainer.callback_metrics:
                        report_dict[key] = trainer.callback_metrics[metric].item()
                    else:
                        logger.warning(f"Metric {metric} does not exist in "
                                       "`trainer.callback_metrics.")

            return report_dict

        def _handle(self, trainer, pl_module):
            report_dict = self._get_report_dict(trainer, pl_module)
            if report_dict is not None:
                tune.report(**report_dict)

    def deepCinet_tune(self, config):
        hdict = vars(self.args)
        hparams = argparse.Namespace(**hdict)

        gene_data = cinet.Dataset(self.dataSet, False)
        train_idx, val_idx = train_test_split(list(range(gene_data.__len__())), test_size=0.2)

        train_dl = cinet.Create_Dataloader(
            cinet.Dataset(self.dataSet, True, delta=0, idxs=train_idx),
            hparams, shuffle_ind=True)
        val_dl = cinet.Create_Dataloader(
            cinet.Dataset(self.dataSet, True, delta=0, idxs=val_idx),  # is_train = true here to get pairs
            hparams, shuffle_ind=True)

        siamese_model = DeepCINET(hparams=hparams, config=config)
        trainer = Trainer(min_epochs=hparams.min_epochs,
                          max_epochs=hparams.max_epochs,
                          min_steps=hparams.min_steps,
                          max_steps=hparams.max_steps,
                          gpus=1,
                          accumulate_grad_batches=hparams.accumulate_grad_batches,
                          # distributed_backend='dp',
                          weights_summary='full',
                          # enable_benchmark=False,
                          num_sanity_val_steps=0,
                          # auto_find_lr=hparams.auto_find_lr,
                          callbacks=[EarlyStopping(monitor='val_ci', patience=10, mode="max"),
                                     cinet.TuneReportCallback({
                                         "best_loss": "best_loss",
                                         "CI": "best_val_ci"
                                     }, on="validation_end")],
                          check_val_every_n_epoch=hparams.check_val_every_n_epoch)
        # overfit_pct=hparams.overfit_pct)
        trainer.fit(siamese_model,
                    train_dataloader=train_dl,
                    val_dataloaders=val_dl)

    def tune_DeepCINET_bohb(self, num_samples=1000):
        hdict = vars(self.args)
        hparams = argparse.Namespace(**hdict)
        gene_data = cinet.Dataset(self.dataSet, False)
        config = {
            "hidden_one": tune.choice([32, 128, 512]),
            "hidden_two": tune.choice([32, 128, 512]),
            "hidden_three": tune.choice([0, 32, 128, 512]),
            "hidden_four": tune.choice([0, 32, 128, 512]),
            "dropout": tune.uniform(0, 0.4),
            "lr": tune.loguniform(1e-4, 1e-1),
            "batchnorm": tune.choice([True, False]),
            "dat_size": gene_data.gene_num()
        }

        algo = TuneBOHB()  # space=config_space,
        # max_concurrent=20)
        bohb = HyperBandForBOHB(time_attr="training_iteration")

        reporter = CLIReporter(
            parameter_columns=["hidden_one", "hidden_two", "hidden_three", "hidden_four", "dropout", "lr", "batchnorm"],
            metric_columns=["best_loss", "CI", "training_iteration"])

        analysis = tune.run(
            tune.with_parameters(
                self.deepCinet_tune),
            resources_per_trial={
                "cpu": 0,
                "gpu": 1
            },
            metric="CI",
            mode="max",
            config=config,
            num_samples=num_samples,
            scheduler=bohb,
            search_alg=algo,
            progress_reporter=reporter,
            name="tune_DeepCINET_bohb_Vorinostat_rnaseq_best_ci",
            local_dir="./HyperparameterTuning/DeepCINET/rnaseq/")

        print("Best hyperparameters found were: ", analysis.best_config)
        print("Results were: ", analysis.best_result)

    def Create_Dataloader(self, ds, hparams, shuffle_ind):
        return torch.utils.data.DataLoader(ds,
                                           batch_size=hparams.batch_size,
                                           shuffle=shuffle_ind,
                                           num_workers=hparams.num_workers)


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
        self.layers_size = [i for i in
                            [self.data_sz, self.hidden_one, self.hidden_two, self.hidden_three, self.hidden_four, 1] if
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
            self.fc = cinet.FullyConnectedLinear(self.layers_size, self.dropout, self.batchnorm)
        else:
            self.fc = cinet.FullyConnected(self.layers_size, self.dropout, self.batchnorm)
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

        # TODO: This does not work, as lightning does not update the
        # drug_response = np.concatenate([x['Drug_response'] for x in outputs])
        # drug_response_pred = np.concatenate([x['Drug_response_pred'] for x in outputs])
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
            milestones=self.hparams.sc_milestones,
            gamma=self.hparams.sc_gamma)

        return [optimizer], [scheduler]

    def log_model_parameters(self):
        print("PARAMETERS**********************************************")
        print("Convolution layer parameters: %d" % (cinet.count_parameters(self.convolution)))
        print("FC layer parameters: %d" % (cinet.count_parameters(self.fc)))
        print("********************************************************")

    @staticmethod
    def add_model_specific_args(ref, parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        ## NETWORK
        # parser.add_argument('--fc-layers', type=int, nargs='+',
        #                     default=default_config.FC_LAYERS)
        # parser.add_argument('--dropout', type=float, nargs='+',
        #                     default=default_config.DROPOUT)

        # parser.add_argument('--use-distance', action='store_true', default=config.USE_DISTANCE)
        # parser.add_argument('--d-layers', type=int, nargs='+', default=config.D_LAYERS)
        # parser.add_argument('--d-dropout', type=float, nargs='+',
        #                     default=[])

        # parser.add_argument('--use-images', action='store_true', default=config.USE_IMAGES)
        # parser.add_argument('--conv-layers', type=int, nargs='+', default=[1, 4, 8, 16])
        # parser.add_argument('--conv-model', type=str, default="Bottleneck")
        # parser.add_argument('--pool', type=int, nargs='+', default=[1, 1, 1, 1])
        ## OPTIMIZER
        # parser.add_argument('--learning-rate', type=float, default=default_config.LR)
        parser.add_argument('--momentum', type=float, default=ref.momentum)
        parser.add_argument('--weight-decay', type=float, default=ref.weight_decay)
        parser.add_argument('--sc-milestones', type=int, nargs='+',
                            default=ref.sc_milestones)
        parser.add_argument('--sc-gamma', type=float, default=ref.sc_gamma)
        # parser.add_argument('--use-exp', action='store_true', default=config.USE_IMAGES)
        return parser