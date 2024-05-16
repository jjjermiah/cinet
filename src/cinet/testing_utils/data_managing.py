# Marc Delgado Sánchez // 6th February 2024 // DeepCINET Project // BHK Lab // University Health Network

# This file contains all functions necessary to transform the csv files extracted
# from the R PSet objects in Orcestra into csv files with the correct format to 
# input in DeepCINET. We are currently (September 2023 - May 2024) using CCLE, 
# gCSI and GDSC-v2 as datasets with AAC values and Genomics (RNA-Seq) data.
# From these datasets, we extracted the expression and drug response values from the PSets
# using an R code (ask Farnoosh). We then use those csv files as input in this code.
#
# Also, due to the low number of drugs that CCLE has, we are using CTRP to obtain drug response values.
# As CTRP does not have genomics data, we are using the one from CCLE and merging the data from both datasets.
# There are functions in this code dedicated to that, as well.
#
# There's also some functions used to prepare the data from clinical trials to input in DeepCINET.

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib_venn as venn

# This function prints and returns all the drugs that are common in the datasets specified in the
# datasets variable. It is important that we train and test on the same drugs across the training
# and testing datasets. Datasets: "gCSI", "CCLE", "GDSC-v2"
def find_common_drugs():
    path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/PSets/csv/"
    datasets = ["gCSI", "GDSC-v2"]
    drugs = []
    for dataset in datasets:
        aac_path = path + dataset + "-aac.csv"
        aac_table = pd.read_csv(aac_path)
        drugs.append(aac_table.iloc[:,0])
    common_drugs = drugs[0]
    idx = 1
    num_datasets = len(datasets)
    ctrp_drugs = pd.DataFrame({"Unnamed: 0": obtain_ctrp_drugs()})
    venn.venn3(subsets=[set(drugs[0]), set(drugs[1]), set(ctrp_drugs.iloc[:,0])], set_labels=["gCSI", "GDSC-v2", "CTRP"])
    plt.title("Common drugs across datasets")
    plt.show()
    while idx < num_datasets:
        common_drugs = pd.merge(common_drugs, drugs[idx])
        idx = idx + 1
    # print(common_drugs.merge(ctrp_drugs))
    return common_drugs.merge(ctrp_drugs)

# This function returns the cells that are common in the datasets specified in the datasets variable.
# Being honest this function is not really necessary for anything, since it is not necessary to have the same
# cell-lines across datasets for training nor testing. Datasets: "gCSI", "CCLE", "GDSC-v2"
def find_common_cells():
    path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/PSets/csv/"
    datasets = ["gCSI", "GDSC-v2"]
    cells = []
    for dataset in datasets:
        expr_path = path + dataset + "-expr.csv"
        expr_cells = pd.read_csv(expr_path).iloc[:,0].unique()
        cells.append(expr_cells)
    ctrp_path = path + "CTRP-cells.csv"
    ctrp_cells = pd.read_csv(ctrp_path).iloc[:,0].unique()
    venn.venn3(subsets=[set(cells[0]), set(cells[1]), set(ctrp_cells)], set_labels=["gCSI", "GDSC-v2", "CTRP"])
    plt.title("Common cell-lines across datasets")
    plt.show()
    result = cells[0]
    idx = 1
    while idx < len(datasets):
        result = set(result).intersection(cells[idx])
        idx = idx + 1
    return sorted(result)

# This function is used to assess the standard deviation of gene expression across cell-lines in a particular dataset.
# The reason for its existance is that, in DeepCINET, there's a process of standardization ((data - mean)/std) that can
# arise errors if the std value is 0. This used to happen at some point during development, but this function is currently unused.
def assess_dataset():
    path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/PSets/Curated Data/CCLE/"
    file =  "Crizotinib.csv"
    whole_path = path + file
    table = pd.read_csv(whole_path)
    table = table.set_index(list(table.columns[[0]]))
    print(table)
    number_of_genes = table.iloc[0].size
    number_of_cell_lines = table.iloc[:,0].size
    sample_gene = table.iloc[:,2]
    sds = np.std(table, axis=0)
    print(sds)
    count = 0
    for sd in sds:
        if sd == 0:
            count = count + 1
    new_count = 0
    for expr in sample_gene:
        if expr < -9.8:
            new_count = new_count + 1
    print(str(count) + "/" + str(number_of_genes) + " genes have 0 standard deviation.")
    print("There are " + str(number_of_cell_lines) + " cell-lines in this dataset.")
    print(str(new_count) + "/" + str(number_of_cell_lines) + " cell-lines have a gene expression value lower than -9.8 for the sampled gene.")

# This function is used to select the common genes across the training (CCLE) dataset and the clinical trial dataset.
# The resulting filtered clinical dataset is saved after keeping only the common genes.
def clinical_filter_genes():
    expr_path = "C:/Users/marcd/Downloads/GSE15622_SE_expr.csv"
    ccle_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/train_data/gene_CCLE_rnaseq_Paclitaxel_response.csv"
    expr_table = pd.read_csv(expr_path)
    expr_table = expr_table.set_index(list(expr_table.columns[[0]]))
    ccle_table = pd.read_csv(ccle_path)

    ccle_genes = ccle_table.columns[2:]
    expr_genes = expr_table.index
    common_genes = list(set(ccle_genes).intersection(expr_genes))
    filtered_expr = expr_table.loc[common_genes].transpose()
    filtered_expr = filtered_expr.reindex(sorted(filtered_expr.columns), axis=1)
    filtered_ccle = ccle_table.loc[:, common_genes]
    filtered_ccle = filtered_ccle.reindex(sorted(filtered_ccle.columns), axis=1)
    filtered_ccle = ccle_table.iloc[:,0:2].join(filtered_ccle)
    print(filtered_ccle)
    filtered_ccle.to_csv("C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/Code/cinet/train_data/gene_CCLE_rnaseq_Paclitaxel-Clinical-15622_response.csv")
    # filtered_expr.index = filtered_expr.index.str.replace("exprs.", "")
    # filtered_expr.to_csv("C:/Users/marcd/Downloads/GSE41998_SE_filtered_expr_.csv")

# This function is used to transform the categorical response of patients into numerical values
# in clinical datasets. The clinical datasets we have been using have the categories:
# "stable disease", "partial response" and "complete response". The first one is assigned a value of 0, 
# since the patients did not respond to the treatment (and a low AAC value is related to low response), while
# the other two are assigned a value of 1. 
def clinical_filter_patients():
    resp_path = "C:/Users/marcd/Downloads/GSE41998_SE_resp.csv"
    expr_path = "C:/Users/marcd/Downloads/GSE41998_SE_filtered_expr_.csv"
    resp_table = pd.read_csv(resp_path).iloc[:,0].str.split("\\t",expand=True).iloc[:,1:]
    resp_table = resp_table.set_index(list(resp_table.columns[[0]]))
    expr_table = pd.read_csv(expr_path)
    expr_table = expr_table.set_index(list(expr_table.columns[[0]]))
    resp_patients = resp_table.index
    filtered_expr = expr_table.loc[resp_patients]
    for idx, elem in enumerate(resp_table.iloc[:,0]):
        if elem == "stable disease":
            resp_table.iloc[idx,0] = 0
        elif elem == "partial response" or elem == "complete response":
            resp_table.iloc[idx,0] = 1
    result = resp_table.join(filtered_expr)
    result.to_csv("C:/Users/marcd/Downloads/GSE41998_SE_final.csv")
    print(result)

# This function is used to obtain the drugs that were tested in the CTRP dataset.
def obtain_ctrp_drugs():
    exps_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/PSets/csv/CTRP-exps.csv"
    exps_table = pd.read_csv(exps_path)
    tested_drugs = sorted(exps_table.iloc[:, 3].unique())
    # print(tested_drugs)
    return tested_drugs

# This function is used to generate suitable csv files to input in DeepCINET from the CTRP experiments file.
# For all drugs common in the testing datasets (gCSI and GDSC-v2), we iterate through the CTRP experiments file
# and select the cells and aac values for whom we have data, and we create a Dataframe object which we save
# as a csv file. After execution, we will have generated one csv file per common drug, with the names of the cell-lines 
# and their aac values to that drug, in the CTRP experiments.
def generate_ctrp_files(common_drugs):
    exps_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/PSets/csv/CTRP-exps.csv"
    exps_table = pd.read_csv(exps_path)
    for drug in common_drugs:
        row = 1
        cells = []
        aac_values = []
        while row < len(exps_table):
            exp_drug = exps_table.iloc[row,3]
            aac = exps_table.iloc[row,10]
            if exp_drug == drug and aac is not None and not pd.isna(aac):
                cells.append(exps_table.iloc[row,2])
                aac_values.append(aac)
            row = row + 1
        indexes = np.argsort(cells)
        cells = sorted(cells)
        aac_values = [aac_values[idx] for idx in indexes]
        dic = {'cell_line': cells, 'target': aac_values}
        dataframe = pd.DataFrame(dic)
        print(dataframe)
        dataframe.to_csv("C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/PSets/Curated Data/CTRP/" + drug + ".csv")

# This function is used to obtain the genes common across the training (CCLE) and testing (gCSI and GDSC-v1) datasets.
# The three datasets we used ended up sharing all genes, but if your datasets have different genes, you should make sure
# that the files used to train DeepCINET models have the same genes as the testing ones.
def find_common_genes():
    path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/PSets/csv/"
    gcsi_path = path + "gCSI-expr-fixed.csv"
    gdsc_path = path + "GDSC-v2-expr-fixed.csv"
    ccle_path = path + "CCLE-expr-fixed.csv"
    gcsi_genes = pd.read_csv(gcsi_path).columns.unique()
    gdsc_genes = pd.read_csv(gdsc_path).columns.unique()
    ccle_genes = pd.read_csv(ccle_path).columns.unique()
    venn.venn3(subsets=[set(gcsi_genes), set(gdsc_genes), set(ccle_genes)], set_labels=["gCSI", "GDSC-v2", "CTRP"])
    plt.title("Common genes across datasets")
    plt.show()
    aux = set(gcsi_genes).intersection(gdsc_genes)
    result = aux.intersection(ccle_genes)
    cgc_genes = extract_cancer_gene_census()
    venn.venn2(subsets=[set(cgc_genes), set(ccle_genes)], set_labels=["Cancer Gene Census", "Datasets"])
    plt.title("Common genes between Cancer Gene Census and our datasets")
    plt.show()
    # print(" gCSI has: " + str(len(gcsi_genes)) + " genes")
    # print(" GDSC has: " + str(len(gdsc_genes)) + " genes")
    # print(" CCLE has: " + str(len(ccle_genes)) + " genes")
    return result

# This function is used to add the genomic data (RNA-Seq) from the CCLE Dataset to the CTRP csv files
# generated in the generate_ctrp_files() function. For each of the files generated with the previous
# function, we merge the genetic information from CCLE, selecting only the cell-lines present in both datasets.
# The resulting file is saved as a csv, which is in the correct format to input in DeepCINET models.
def add_ctrp_genes():
    path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/PSets/Curated Data/CTRP/"
    ccle_file = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/PSets/csv/CCLE-expr.csv"
    save_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/PSets/Curated Data/CTRP-expr/"
    ccle_table = pd.read_csv(ccle_file)
    ccle_table = ccle_table.set_index(list(ccle_table.columns[[0]]))
    print(ccle_table.index)
    for filename in os.listdir(path):
        idx = 0
        drug_name = ""
        while filename[idx] != '.':
            drug_name = drug_name + filename[idx]
            idx = idx + 1
        ctrp_table = pd.read_csv(path + filename)
        ctrp_table = ctrp_table.set_index(list(ctrp_table.columns[[1]]))
        result = ctrp_table.iloc[:,1:].merge(ccle_table, left_index=True, right_index=True, sort=True)
        print(result)
        result.to_csv(save_path + filename)

# This function is used to extract the "ENSG" names from the genes in the Cancer Gene Census csv file downloaded from
# https://cancer.sanger.ac.uk/census . 
def extract_cancer_gene_census():
    path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/PSets/csv/CancerGeneCensus.csv"
    table = pd.read_csv(path)
    gene_names = table["Synonyms"].str.split(',',expand=True)
    columns = gene_names.columns.tolist()
    genes = []
    for _, idx in gene_names.iterrows():
        for c in columns:
            value = idx[c]
            if isinstance(value, str) and value[0:3] == "ENS":
                genes.append(value)
    genes = sorted(genes)
    new_genes = []
    for gene in genes:
        idx = 0
        char = gene[idx]
        length = len(gene)
        new_c = ""
        while idx < length and char != '.':
            new_c = new_c + char
            idx = idx + 1
            char = gene[idx]
        new_genes.append(new_c)
    return new_genes

# There are 733 common genes between the datasets and the CGC.
def generate_cancer_gene_census_files(dataset, genes):
    path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/PSets/Curated Data/" + dataset + "-expr/"
    safe_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/PSets/Curated Data/" + dataset + "/"
    for filename in os.listdir(path):
        idx = 0
        drug_name = ""
        while filename[idx] != '.':
            drug_name = drug_name + filename[idx]
            idx = idx + 1
        print(drug_name + " (" + dataset + ')')
        table = pd.read_csv(path + filename, index_col=0)
        columns = table.columns.to_list()
        common = list(set(columns).intersection(genes))
        print(common)
        result = pd.DataFrame(table[["target"]]).join(table[common])
        print(result)
        result.to_csv(safe_path + drug_name + "-CGC.csv")

def fix_genes_files(dataset):
    path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/PSets/Curated Data/" + dataset + '-expr/'
    for filename in os.listdir(path):
        print(filename + " (" + dataset + ')')
        table = pd.read_csv(path + filename)
        new_cols = ["cell_line", "target"]
        for c in tqdm(table.columns.to_list()):
            if c[0] == 'E':
                idx = 0
                char = c[idx]
                new_c = ""
                while idx < len(c) and char != '.':
                    new_c = new_c + char
                    idx = idx + 1
                    char = c[idx]
                new_cols.append(new_c)
        table.columns = new_cols
        table.to_csv(path + filename[0:(len(filename)-4)] + "-fixed.csv")

def sort_cell_lines_datasets(dataset, common):
    path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/PSets/Curated Data/" + dataset + '-CGC/'
    for filename in os.listdir(path):
        print(filename + " (" + dataset + ')')
        table = pd.read_csv(path + filename)
        new_table = table.sort_values('cell_line')
        new_table = pd.DataFrame(new_table[["cell_line","target"]]).join(new_table[common])
        print(new_table)
        new_table.to_csv(path + filename)

# clinical_filter_genes()
# clinical_filter_patients()

# There's no need for cells to be common across datasets!
# Always use GDSC-v2 !!
# drug_list = list(find_common_drugs()["Unnamed: 0"])
# generate_ctrp_files(drug_list)
        
# common_genes = find_common_genes()
# print("There are: " + str(len(common_genes)) + " common genes.")
# add_ctrp_genes()
# genes = extract_cancer_gene_census()
# fix_genes_files("CTRP")
# generate_cancer_gene_census_files("CTRP", new_genes)
        
print("Hi")
# find_common_drugs()
# find_common_cells()
find_common_genes()

"""
dataset = "GDSC-v2"
genes = extract_cancer_gene_census()
new_genes = []
for gene in genes:
    idx = 0
    char = gene[idx]
    length = len(gene)
    new_c = ""
    while idx < length and char != '.':
        new_c = new_c + char
        idx = idx + 1
        char = gene[idx]
    new_genes.append(new_c)
datasets = ["CTRP", "gCSI", "GDSC-v2"]
print(new_genes)
for dataset in datasets:
    # generate_cancer_gene_census_files(dataset, new_genes)
    path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/PSets/Curated Data/" + dataset + "/"
    table = pd.read_csv(path + "5-Fluorouracil-fixed.csv")
    columns = table.columns.to_list()
    common = list(set(columns).intersection(new_genes))
    sort_cell_lines_datasets(dataset, common)
# print("There are " + str(len(genes)) + " genes in the Cancer Gene Census.")
# generate_cancer_gene_census_files(dataset, genes)

# datasets = ["gCSI", "GDSC-v1"]
# for dataset in datasets:
#   fix_genes_files(dataset)
"""

"""
drug_list = list(find_common_drugs()["Unnamed: 0"])
path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/PSets/csv/"
dataset = "GDSC-v2"
aac_path = path + dataset + "-aac.csv"
expr_path = path + dataset + "-expr-fixed.csv"
aac_table = pd.read_csv(aac_path)
expr_table = pd.read_csv(expr_path, index_col=[1]).iloc[:,1:].sort_index()
row = 0
drugs = aac_table.iloc[:,0]
num_drugs = len(drugs)
print("Number of drugs: " + str(num_drugs))
cgc_genes = extract_cancer_gene_census()
new_genes = []
for gene in cgc_genes:
    idx = 0
    char = gene[idx]
    length = len(gene)
    new_c = ""
    while idx < length and char != '.':
        new_c = new_c + char
        idx = idx + 1
        char = gene[idx]
    new_genes.append(new_c)
while row < num_drugs:
    drug_name = drugs.iloc[row]
    if drug_name in drug_list:
        print(drug_name)
        response = aac_table.iloc[row,1:][aac_table.iloc[row, 1:].notnull()]        # Extract not null drug response values for a certain drug.
        data = {'target': response}
        df = pd.DataFrame(data)
        cell_lines = df.index                                                       # Get these cell-lines names
        present_cell_lines = df.index.isin(expr_table.index)                        # Intersect these cell-lines with the ones in the gene expression file.
        df = df.loc[present_cell_lines]
        expr_data = expr_table.loc[expr_table.index.intersection(cell_lines)]       # Extract the gene expression values of the cell-lines for whom we have drug response values.
        df = df.join(expr_data)                                                     # Add the gene expression values to the drug response table
        if '/' in drug_name:
            drug_name = drug_name.replace("/","-")
        columns = df.columns.to_list()
        common = list(set(columns).intersection(new_genes))
        result = pd.DataFrame(df[["target"]]).join(df[common])
        save_path = "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/PSets/Curated Data/" + dataset + "/" + drug_name + ".csv"
        result.to_csv(save_path)
        print("Row number: " + str(row) + "/" + str(num_drugs))
    row = row + 1
"""
"""
    CI = 0.5116882712687322 # When training 5 folds, 10 times 10 epochs each

    Validation CI for 20 epochs training per fold: [0.5013382922768086, 0.5277695647437792, 0.49943200385926145, 0.4964908731579029, 0.49579838471234494, 0.4812405658175254, 0.5231477879273587, 0.49717558083440966, 0.5220896033363938, 0.4997743576975148]
"""