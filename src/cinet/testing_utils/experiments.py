from cinet import *
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from pymrmre import mrmr_ensemble
from lifelines.utils import concordance_index
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

def train(drug, delta, batch_size, epochs, version, arch, cross_validation):
    path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/train_data/"
    # path = "/home/marc_delgado_sanchez_uhn_ca/train_data/"
    file = drug + ".csv"
    whole_path = path + file
    table = pd.read_csv(whole_path)
    table = table.set_index(list(table.columns[[0]]))
    print(table)
    X = table.iloc[:,1:] # X contains all genomic information
    y = table.iloc[:,0]  # y contains the response data (AAC)
    # print(X)
    # print(y)
    param = {'delta': delta, 'batch_size': batch_size, 'max_epochs': epochs, 'architecture': arch}

    device = 'cpu'
    model = deepCINET(device=device,delta=delta, batch_size=batch_size, max_epochs=epochs, nnHiddenLayers=arch)
    val_ci = model.fit(X,y, cross_validation)
    param_json = json.dumps(param)
    json_file_name = "params/" + drug + "-" + version +"-param.json"
    with open(json_file_name, "w") as outfile:
        outfile.write(param_json)
    
    model_file_name = "models/" + drug + "-" + version + ".pth"
    torch.save(model, model_file_name)
    return val_ci

def test_gcsi(drug, version):
    path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/test_data/gCSI_Test_Data/"
    # path = "/home/marc_delgado_sanchez_uhn_ca/cinet/test_data/gCSI_Test_Data/"
    file = drug + ".csv"
    whole_path = path + file

    table = pd.read_csv(whole_path).set_index('cell_line')
    X = table.iloc[:,1:] # X contains all genomic information
    y = table.iloc[:,0]  # y contains the response data (AAC)

    model_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/models/"
    # model_path = "/home/marc_delgado_sanchez_uhn_ca/cinet/models/"
    model_name = drug + "-" + version + ".pth"
    whole_model_path = model_path + model_name
    model = torch.load(whole_model_path)

    concordance = model.score(X, y)
    return concordance

def test_gdsc(drug, version):
    path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/test_data/GDSC_Test_Data/"
    # path = "/home/marc_delgado_sanchez_uhn_ca/cinet/test_data/GDSC_Test_Data/"
    file = drug + ".csv"
    whole_path = path + file

    table = pd.read_csv(whole_path).set_index('cell_line')
    X = table.iloc[:,1:] # X contains all genomic information
    y = table.iloc[:,0]  # y contains the response data (AAC)

    model_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/models/"
    # model_path = "/home/marc_delgado_sanchez_uhn_ca/cinet/models/"
    model_name = drug + "-" + version + ".pth"
    whole_model_path = model_path + model_name
    model = torch.load(whole_model_path)

    concordance = model.score(X, y)
    return concordance

def test_ci(versions, drugs):
    gen_gcsi = []
    gen_gdsc = []
    for drug_name in drugs:
        gcsi = []
        gdsc = []
        for vers in versions:
            ci_gcsi = test_gcsi(drug_name, vers)
            ci_gdsc = test_gdsc(drug_name, vers)
            gcsi.append(ci_gcsi)
            gdsc.append(ci_gdsc)
        gen_gcsi.append(gcsi)
        gen_gdsc.append(gdsc)
    return (gen_gcsi, gen_gdsc)

def test_clinical(version, cohort, drug):
    path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/clinical_data/"
    file = "GSE" + cohort + "_SE_final.csv"
    whole_path = path + file
    table = pd.read_csv(whole_path)
    table = table.set_index(list(table.columns[[0]]))
    X = table.iloc[:,1:] # X contains all genomic information
    y = table.iloc[:,0]  # y contains the response data (AAC)

    # print(X)
    # print(y)

    model_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/models/DSI Models/"  
    model_name = drug + "-Clinical-" + cohort + '-' + version + ".pth"
    whole_model_path = model_path + model_name
    model = torch.load(whole_model_path)

    preds = model.predict(X)
    concordance = concordance_index(y.tolist(), preds.tolist())
    return concordance

def plot_ci_delta(versions, drugs):
    results = []
    for drug_name in drugs:
        gcsi = []
        gdsc = []
        for vers in versions:
            ci_gcsi = test_gcsi(drug_name, vers)
            ci_gdsc = test_gdsc(drug_name, vers)
            gcsi.append(ci_gcsi)
            gdsc.append(ci_gdsc)
        results.append((gcsi, gdsc))
    return results

def mass_train():
    other_deltas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    deltas = np.linspace(0, 0.1, 11)
    train_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/train_data/"
    # train_path = "/home/marc_delgado_sanchez_uhn_ca/cinet/train_data"
    batch_size = 128
    max_epochs = 20
    count = 0
    for filename in os.listdir(train_path):
        drug_name = ""
        idx = 17
        print(filename)
        while filename[idx] != "_":
            drug_name = drug_name + filename[idx]
            idx = idx + 1
        if count > 0:
            vers = 0
            for delta in deltas:
                train(drug_name, delta, batch_size, max_epochs, str(vers))
                vers = vers + 1
        else:
            vers = 1
            count = count + 1
            for delta in other_deltas:
                train(drug_name, delta, batch_size, max_epochs, str(vers))
                vers = vers + 1

def architecture_exploration():
    experiment_drugs = ["AZD7762", "Dabrafenib", "Ibrutinib", "Lapatinib", "Pictilisib", "Vorinostat"]
    batch_size = 128
    epochs = 1
    cross_validation_results = []
    architectures = [(128,128,0,0), (128,256,128,0), (128,512,128,0), (128,256,256,128), (128,512,512,128)]
    deltas = np.append(np.linspace(0, 0.1, 11), [0.15, 0.2])
    parameters = {'drugs': experiment_drugs, 'architectures': architectures, 'deltas': deltas, 'batch_size': batch_size,
                  'epochs': epochs}
    num_exp = len(experiment_drugs)*len(architectures)*len(deltas)
    count = 1
    for arch in architectures:
        arch_ci = []
        for drug in experiment_drugs:
            drug_ci = []
            vers = 0
            for delta in deltas:
                val_ci = train(drug, delta, batch_size, epochs, version=str(vers), arch=arch, cross_validation=True)
                vers += 1
                drug_ci.append(np.average(val_ci))
                print("Experiment " + str(count) + '/' + str(num_exp))
            arch_ci.append(drug_ci)
        cross_validation_results.append(arch_ci)
    np.save("ArchitectureExperimentResults.npy", cross_validation_results)
    param_json = json.dumps(parameters)
    json_file_name = "ArchitectureExperimentParameters.json"
    with open(json_file_name, "w") as outfile:
        outfile.write(param_json)
    return cross_validation_results

def hyperparameter_exploration():
    batch_sizes = [64, 128, 256]
    dropout_rates = [0.2, 0.4, 0.5]
    learning_rates = [0.1, 0.01, 0.001]

    experiment_drugs = ["Dabrafenib", "Lapatinib", "Pictilisib", "Vorinostat"] # To be changed accordingly
    arch = () # To be filled with the architecture that shows best results in previous experiment
    epochs = 20 # To be filled with the number of epochs that shows best results in previous experiment
    deltas = np.linspace(0,0.2,5) # To be changed accordingly
    parameters = {'drugs': experiment_drugs, 'architecture': arch, 'deltas': deltas, 'batch_sizes': batch_sizes,
                  'epochs': epochs, 'dropout_rates': dropout_rates, "learning_rates": learning_rates}
    
    experiment_result = []
    for batch_size in batch_sizes:
        batch_result = []
        for dropout_rate in dropout_rates:
            dropout_result = []
            for learning_rate in learning_rates:
                lr_result = []
                for drug in experiment_drugs:
                    vers = 0
                    drug_result = []
                    for delta in deltas:
                        val_ci = train(drug, delta, batch_size, epochs, version=str(vers), arch=arch,
                                        dropout = dropout_rate, learning_rate = learning_rate, cross_validation=True)
                        drug_result.append(val_ci)
                        vers += 1
                    lr_result.append(drug_result)
                dropout_result.append(lr_result)
            batch_result.append(dropout_result)
        experiment_result.append(batch_result)

    np.save("HyperparameterExperimentResults.npy", experiment_result)
    param_json = json.dumps(parameters)
    json_file_name = "HyperparameterExperimentParameters.json"
    with open(json_file_name, "w") as outfile:
        outfile.write(param_json)
    return experiment_result

def method_validation():
    drugs = ["AZD7762", "Lapatinib", "Vorinostat"]
    deltas = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2]
    results = []
    # path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/train_data/"
    path = "/home/marc_delgado_sanchez_uhn_ca/train_data/"
    device = 'cpu'
    batch_size = 32
    epochs = 10
    arch = (128, 256, 128, 0)
    cross_validation = False
    random_pairs = True
    for drug in drugs:
        drug_result = []
        file = drug + ".csv"
        whole_path = path + file
        table = pd.read_csv(whole_path)
        table = table.set_index(list(table.columns[[0]]))
        X = table.iloc[:,1:] # X contains all genomic information
        y = pd.DataFrame({'target': table.iloc[:,0]})  # y contains the response data (AAC)
        selected_genes = mrmr_ensemble(X, y, 100)
        new_X = X[selected_genes[0][0]]
        for delta in deltas:
            model = deepCINET(device=device,delta=delta, batch_size=batch_size, max_epochs=epochs, nnHiddenLayers=arch)
            scores = model.fit(new_X,y, cross_validation, random_pairs)
            drug_result.append(scores)
        results.append(drug_result)
        np.save("Method-Validation-" + drug + "-Results.npy", drug_result)
    print(results)

def model_inference():
    drugs = ["Lapatinib", "Vorinostat", "Crizotinib", "Dabrafenib", "Docetaxel"]
    deltas = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2]
    results = []
    # gcsi_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/test_data/gCSI_Test_Data/"
    # gdsc_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/test_data/GDSC_Test_Data/"
    gcsi_path = "/home/marc_delgado_sanchez_uhn_ca/test_data/gCSI_Test_Data/"
    gdsc_path = "/home/marc_delgado_sanchez_uhn_ca/test_data/GDSC_Test_Data/"
    device = 'cpu'
    batch_size = 32
    epochs = 15
    arch = (128, 256, 128, 0)
    cross_validation = False
    random_pairs = False
    device = 'cpu'
    for drug in drugs:
        drug_result = []
        file = drug + ".csv"
        gdsc_table = pd.read_csv(gdsc_path + file)
        gcsi_table = pd.read_csv(gcsi_path + file)
        gdsc_table = gdsc_table.set_index(list(gdsc_table.columns[[0]]))
        gcsi_table = gcsi_table.set_index(list(gcsi_table.columns[[0]]))
        common_cells = set.intersection(set(gdsc_table.index), set(gcsi_table.index))
        other_cells_gcsi = set(gcsi_table.index) - common_cells
        other_cells_gdsc = set(gdsc_table.index) - common_cells

        X_gcsi = gcsi_table.iloc[:,1:]
        y_gcsi = pd.DataFrame(data={'target': gcsi_table.iloc[:,0]})
        selected_genes = mrmr_ensemble(X_gcsi, y_gcsi, 100)
        X_gcsi = X_gcsi[selected_genes[0][0]]

        X_common_gdsc = gdsc_table.loc[common_cells].iloc[:,1:][selected_genes[0][0]]
        y_common_gdsc = pd.DataFrame(data={'target': gdsc_table.loc[common_cells].iloc[:,0]})
        X_diff_gdsc = gdsc_table.loc[other_cells_gdsc].iloc[:,1:][selected_genes[0][0]]
        y_diff_gdsc = pd.DataFrame(data={'target': gdsc_table.loc[other_cells_gdsc].iloc[:,0]})

        print(drug)
        print("Common cells: " + str(len(common_cells)))
        print("Other gCSI cells: " + str(len(other_cells_gcsi)))
        print("Other GDSC cells: " + str(len(other_cells_gdsc)))

        for delta in deltas:
            model = deepCINET(device=device,delta=delta, batch_size=batch_size, max_epochs=epochs, nnHiddenLayers=arch)
            val_ci = model.fit(X_gcsi, y_gcsi, cross_validation, random_pairs)
            if val_ci != -2:
                common_predictions = model.predict(X_common_gdsc)
                common_score = concordance_index(y_common_gdsc, common_predictions)
                diff_predictions = model.predict(X_diff_gdsc)
                diff_score = concordance_index(y_diff_gdsc, diff_predictions)
            else:
                common_score = -1.0
                diff_score = -1.0
            drug_result.append((common_score, diff_score))
        np.save("Model-Inference-" + drug + "-Results.npy", drug_result)
        results.append(drug_result)
    print(results)

def correlation_exploration():
    deltas = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2]
    drugs = ["Dabrafenib", "Gemcitabine", "Erlotinib", "AZD8055", "Lapatinib", "AZD7762"]
    epochs = 1
    batch_size = 32
    num_genes = 100
    train_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/train_data/"
    # train_path = "/home/marc_delgado_sanchez_uhn_ca/train_data/"
    gcsi_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/test_data/gCSI_Test_Data/"
    gdsc_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/test_data/GDSC_Test_Data/"
    # gcsi_path = "/home/marc_delgado_sanchez_uhn_ca/test_data/gCSI_Test_Data/"
    # gdsc_path = "/home/marc_delgado_sanchez_uhn_ca/test_data/GDSC_Test_Data/"
    arch = (128, 256, 128, 0)
    cross_validation = False
    random_pairs = False
    device = 'cpu'
    results = []
    for drug in drugs:
        drug_result = []
        file = drug + ".csv"
        train_table = pd.read_csv(train_path + file)
        gdsc_table = pd.read_csv(gdsc_path + file)
        gcsi_table = pd.read_csv(gcsi_path + file)
        gdsc_table = gdsc_table.set_index(list(gdsc_table.columns[[0]]))
        gcsi_table = gcsi_table.set_index(list(gcsi_table.columns[[0]]))
        train_table = train_table.set_index(list(train_table.columns[[0]]))

        X = train_table.iloc[:,1:] # X contains all genomic information
        y = pd.DataFrame({'target': train_table.iloc[:,0]})  # y contains the response data (AAC)
        selected_genes = mrmr_ensemble(X, y, 100)
        new_X = X[selected_genes[0][0]]

        X_gcsi = gcsi_table.iloc[:,1:][selected_genes[0][0]]
        y_gcsi = pd.DataFrame(data={'target': gcsi_table.iloc[:,0]})
        X_gdsc = gdsc_table.iloc[:,1:][selected_genes[0][0]]
        y_gdsc = pd.DataFrame(data={'target': gdsc_table.iloc[:,0]})
        for delta in deltas:
            model = deepCINET(device=device,delta=delta, batch_size=batch_size, max_epochs=epochs, nnHiddenLayers=arch)
            scores = model.fit(new_X,y, cross_validation, random_pairs)
            gcsi_preds = model.predict(X_gcsi)
            gdsc_preds = model.predict(X_gdsc)
            gcsi_score = concordance_index(y_gcsi.iloc[:,0].tolist(), gcsi_preds.tolist())
            gdsc_score = concordance_index(y_gdsc.iloc[:,0].tolist(), gdsc_preds.tolist())
            drug_result.append((gcsi_score, gdsc_score))
        np.save("Correlation-" + drug + ".npy", drug_result)
        results.append(drug_result)
    print(results)

def overfit_exp():
    experiment_drugs = ["AZD7762", "Dabrafenib", "Ibrutinib"]
    # experiment_drugs = ["Lapatinib", "Pictilisib", "Vorinostat"]
    batch_size = 64
    dropout = 0.5
    epochs = 3
    cross_validation_results = []
    architectures = [(64,64,0,0)]
    # architectures = [(128,256,128,0), (128,512,128,0)]
    # architectures = [(128,256,256,128), (128,512,512,128)]
    # deltas = np.append(np.linspace(0, 0.1, 11), [0.15, 0.2])
    deltas = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2]
    parameters = {'drugs': experiment_drugs, 'architectures': architectures, 'deltas': deltas, 'batch_size': batch_size,
                  'epochs': epochs, 'dropout': dropout}
    param_json = json.dumps(parameters)
    json_file_name = "OverfitExperimentParams.json"
    with open(json_file_name, "w") as outfile:
        outfile.write(param_json)
    for arch in architectures:
        arch_ci = []
        for drug in experiment_drugs:
            drug_ci = []
            vers = 0
            for delta in deltas:
                val_ci = train(drug, delta, batch_size, epochs, version=str(vers), arch=arch, dropout=dropout)
                vers += 1
                drug_ci.append(np.average(val_ci))
            arch_ci.append(drug_ci)
            results_file_name = "OverfitExperiment-" + drug + ".npy"
            np.save(results_file_name, drug_ci)
        cross_validation_results.append(arch_ci)
    return cross_validation_results

def less_genes_exp():
    experiment_drugs = ["AZD7762", "Dabrafenib", "Ibrutinib"]
    batch_size = 64
    dropout = 0.5
    epochs = 10
    arch = (64,64,0,0)
    deltas = [0.0, 0.05, 0.1, 0.15, 0.2]
    gene_indices = [50, 100, 200]
    parameters = {'drugs': experiment_drugs, 'architecture': arch, 'deltas': deltas, 'batch_size': batch_size,
                  'epochs': epochs, 'dropout': dropout, 'gene indices': gene_indices}
    param_json = json.dumps(parameters)
    json_file_name = "LessGenesExperiment.json"
    with open(json_file_name, "w") as outfile:
        outfile.write(param_json)
    cross_validation_results = []
    for drug in experiment_drugs:
        drug_ci = []
        for gene_index in gene_indices:
            gene_ci = []
            vers = 0
            for delta in deltas:
                val_ci = train(drug, delta, batch_size, epochs, version=str(vers), arch=arch, dropout=dropout, gene_index=gene_index)
                vers += 1
                drug_ci.append(np.average(val_ci))
            gene_ci.append(drug_ci)
        cross_validation_results.append(gene_ci)
        results_file_name = "LessGenesExperiment-" + drug + ".npy"
        np.save(results_file_name, drug_ci)
    return cross_validation_results

def mrmr_experiment():
    experiment_drugs = ["AZD7762"]
    batch_size = 64
    dropout = 0.5
    epochs = 20
    architectures = [(32,32,0,0), (64,64,0,0), (128,128,0,0)]
    deltas = [0.0, 0.05, 0.1, 0.15, 0.2]
    gene_indices = [50, 100, 200]
    parameters = {'drugs': experiment_drugs, 'architecture': architectures, 'deltas': deltas, 'batch_size': batch_size,
                  'epochs': epochs, 'dropout': dropout, 'gene indices': gene_indices}
    param_json = json.dumps(parameters)
    json_file_name = "mRMRExperiment.json"
    with open(json_file_name, "w") as outfile:
        outfile.write(param_json)
    for drug in experiment_drugs:
        for arch in architectures:
            arch_ci = []
            for gene_index in gene_indices:
                gene_ci = []
                vers = 0
                for delta in deltas:
                    val_ci = train(drug, delta, batch_size, epochs, version=str(vers), arch=arch, dropout=dropout, gene_index=gene_index)
                    vers += 1
                    gene_ci.append(np.average(val_ci))
                arch_ci.append(gene_ci)
            results_file_name = "mRMRExperiment-" + drug + "-" + str(arch) + ".npy"
            np.save(results_file_name, arch_ci)

def learn_experiment():
    experiment_drugs = ["AZD7762", "Lapatinib", "Vorinostat"]
    batch_size = 64
    dropout = 0.5
    epochs = 40
    architectures = [(32,32,0,0), (128,256,128,0)]
    deltas = [0.0, 0.05, 0.1, 0.15, 0.2]
    gene_indices = 100
    parameters = {'drugs': experiment_drugs, 'architecture': architectures, 'deltas': deltas, 'batch_size': batch_size,
                  'epochs': epochs, 'dropout': dropout, 'gene indices': gene_indices}
    param_json = json.dumps(parameters)
    json_file_name = "LearnExperiment.json"
    with open(json_file_name, "w") as outfile:
        outfile.write(param_json)
    for drug in experiment_drugs:
        # test_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/test_data/"
        test_path = "/home/marc_delgado_sanchez_uhn_ca/test_data/"
        # train_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/train_data/" + drug + ".csv"
        train_path = "/home/marc_delgado_sanchez_uhn_ca/train_data/" + drug + ".csv"
        table = pd.read_csv(train_path)
        table = table.set_index(list(table.columns[[0]]))
        X = table.iloc[:,1:] # X contains all genomic information
        y = pd.DataFrame({'target': table.iloc[:,0]})  # y contains the response data (AAC)
        selected_genes = mrmr_ensemble(X, y, gene_indices)
        gcsi = pd.read_csv(test_path + "gCSI_Test_Data/" + drug + ".csv")
        gcsi = gcsi.set_index(list(gcsi.columns[[0]]))
        gdsc = pd.read_csv(test_path + "GDSC_Test_Data/" + drug + ".csv")
        gdsc = gdsc.set_index(list(gdsc.columns[[0]]))
        X_gcsi = gcsi[selected_genes[0][0]]
        y_gcsi = gcsi.iloc[:,0]
        X_gdsc = gdsc[selected_genes[0][0]]
        y_gdsc = gdsc.iloc[:,0]
        for arch in architectures:
            arch_scores = []
            vers = 0
            for delta in deltas:
                model = train(drug, delta, batch_size, epochs, version=str(vers), arch=arch, dropout=dropout, gene_index=gene_indices, cross_validation=False)
                vers += 1
                gcsi_score = model.score(X_gcsi, y_gcsi)
                gdsc_score = model.score(X_gdsc, y_gdsc)
                arch_scores.append((gcsi_score, gdsc_score))
                # Test on gCSI and GDSC
            results_file_name = "LearnExperiment-" + drug + "-" + str(arch) + ".npy"
            np.save(results_file_name, arch_scores)

# ELASTIC NET TESTS

def classify_target(y):
        max_val = max(y)
        min_val = min(y)
        vec_folds = list(range(1, 6))
        dif = (max_val - min_val)/5
        bins = [min_val-0.01]
        for elem in vec_folds:
            bins.append(min_val+dif*elem)
        result = pd.cut(y, bins, labels=vec_folds)
        return result

def train_test_split_plot():
    drugs = ["AZD7762", "Ibrutinib", "Lapatinib", "Pictilisib", "Vorinostat"]
    path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/train_data/"
    indices = np.arange(start=20, stop=700, step=10)
    for drug in drugs:
        file = drug + ".csv"
        whole_path = path + file
        table = pd.read_csv(whole_path)
        table = table.set_index(list(table.columns[[0]]))
        scores = []
        for index in indices:
            X = table.iloc[:,1:index] # X contains all genomic information
            y = table.iloc[:,0]  # y contains the response data (AAC)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred_lr = model.predict(X_test)
            score = concordance_index(y_test, y_pred_lr)
            scores.append(score)
        plt.plot(indices, scores, 'o-')
    plt.grid(True)
    plt.legend(drugs)
    plt.xlabel("Number of genes")
    plt.ylabel("Concordance Index")
    plt.title("Linear Regression with Basic train-test split")
    plt.show()

def cross_validation_plot():
    drugs = ["AZD7762", "Dabrafenib", "Ibrutinib", "Lapatinib", "Pictilisib", "Vorinostat"]
    path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/train_data/"
    indices = np.arange(start=20, stop=700, step=10)
    for drug in drugs:
        file = drug + ".csv"
        whole_path = path + file
        table = pd.read_csv(whole_path)
        table = table.set_index(list(table.columns[[0]]))
        scores = []
        for index in indices:
            X = table.iloc[:,1:index] # X contains all genomic information
            y = table.iloc[:,0]  # y contains the response data (AAC)
            combined_df = pd.concat([X,y],axis=1)
            combined_df.columns.values[-1] = 'target'
            num_folds = 5
            new_y = classify_target(y)
            skf = StratifiedKFold(n_splits=num_folds, random_state=None)
            result = skf.split(X,new_y)
            global_prediction = pd.DataFrame({'target': y})
            first = True
            for train_index, val_index in result:
                elasticNet = LinearRegression()
                train_dataset = X.iloc[train_index]
                train_keys = y.iloc[train_index]
                val_dataset = X.iloc[val_index]
                elasticNet.fit(train_dataset, train_keys)
                predictions = elasticNet.predict(val_dataset)
                if first:
                    global_prediction.insert(1, 'Predictions', np.zeros(len(y)))
                    first = False
                count = 0
                for idx in val_index:
                    global_prediction.iloc[idx, 1] = predictions[count]
                    count += 1
            val_ci = concordance_index(y.tolist(), global_prediction.iloc[:,1].tolist())
            scores.append(val_ci)
        plt.plot(indices, scores, 'o-')
    plt.grid(True)
    plt.legend(drugs)
    plt.xlabel("Number of genes")
    plt.ylabel("Concordance Index")
    plt.title("Linear Regression with Stratified Cross-Validation")
    plt.show()

def cross_val_mrmr():
    drugs = ["AZD7762", "Dabrafenib", "Ibrutinib", "Lapatinib", "Pictilisib", "Vorinostat"]
    path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/train_data/"
    indices = np.arange(start=20, stop=700, step=10)
    for drug in drugs:
        file = drug + ".csv"
        whole_path = path + file
        table = pd.read_csv(whole_path)
        table = table.set_index(list(table.columns[[0]]))
        scores = []
        X = table.iloc[:,1:]
        y = pd.DataFrame({'target': table.iloc[:,0]}) # y contains the response data (AAC)
        num_folds = 5
        new_y = classify_target(y.iloc[:,0])
        skf = StratifiedKFold(n_splits=num_folds, random_state=None)
        for index in indices:
            selected_genes = mrmr_ensemble(X, y, index)
            new_X = X[selected_genes[0][0]]
            result = skf.split(X=new_X,y=new_y)
            global_prediction = pd.DataFrame({'target': y.iloc[:,0]})
            first = True
            for train_index, val_index in result:
                elasticNet = LinearRegression()
                train_dataset = new_X.iloc[train_index]
                train_keys = y.iloc[train_index]
                val_dataset = new_X.iloc[val_index]
                elasticNet.fit(train_dataset, train_keys)
                predictions = elasticNet.predict(val_dataset)
                if first:
                    global_prediction.insert(1, 'Predictions', np.zeros(len(y)))
                    first = False
                count = 0
                for idx in val_index:
                    global_prediction.iloc[idx, 1] = predictions[count]
                    count += 1
            val_ci = concordance_index(y.iloc[:,0].tolist(), global_prediction.iloc[:,1].tolist())
            scores.append(val_ci)
        plt.plot(indices, scores, 'o-')
    plt.grid(True)
    plt.legend(drugs)
    plt.xlabel("Number of genes")
    plt.ylabel("Concordance Index")
    plt.title("Linear Regression with Stratified Cross-Validation on mRMR-selected genes")
    plt.show()

def gcsi_gdsc_tests():
    experiment_drugs = ["AZD7762", "Lapatinib", "Vorinostat"]
    gene_indices = 100
    scores = []
    for drug in experiment_drugs:
        test_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/test_data/"
        train_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/train_data/" + drug + ".csv"
        gcsi = pd.read_csv(test_path + "gCSI_Test_Data/" + drug + ".csv")
        gdsc = pd.read_csv(test_path + "GDSC_Test_Data/" + drug + ".csv")

        gcsi = gcsi.set_index(list(gcsi.columns[[0]]))
        gdsc = gdsc.set_index(list(gdsc.columns[[0]]))
        table = pd.read_csv(train_path)
        table = table.set_index(list(table.columns[[0]]))
        X = table.iloc[:,1:] # X contains all genomic information
        y = pd.DataFrame({'target': table.iloc[:,0]})  # y contains the response data (AAC)

        selected_genes = mrmr_ensemble(X, y, gene_indices)
        X_gcsi = gcsi[selected_genes[0][0]]
        y_gcsi = gcsi.iloc[:,0]
        X_gdsc = gdsc[selected_genes[0][0]]
        y_gdsc = gdsc.iloc[:,0]
        new_X = X[selected_genes[0][0]]

        model = LinearRegression()
        model.fit(new_X, y)
        y_pred_gcsi = model.predict(X_gcsi)
        y_pred_gdsc = model.predict(X_gdsc)
        score_gcsi = concordance_index(y_gcsi, y_pred_gcsi)
        score_gdsc = concordance_index(y_gdsc, y_pred_gdsc)
        scores.append((score_gcsi, score_gdsc))
    print(scores)
    np.save("ElasticNet-gCSI-GDSC.npy", scores)