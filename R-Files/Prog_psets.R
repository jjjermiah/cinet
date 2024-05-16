## load libraries

library(qs)
library(dplyr)
library(PharmacoGx)

## RNA-seq PSets

extractRNASeqData <- function(pset){
  
  # Removing duplications and filtering tissue types -> duplicates have "PAR_Y" at the end of the column name 
  exp_mat <- summarizeMolecularProfiles(pset, mDataType = "Kallisto_0.46.1.rnaseq") %>% assay() %>% t() 
  exp_mat <- exp_mat[ , -grep("PAR_Y", colnames(exp_mat))]
  exp_mat <- exp_mat[rowSums(is.na(exp_mat)) != ncol(exp_mat), ] # Removing rows in which the gene expression is NA for all genes
  
  gene_ann <- as.data.frame(featureInfo(pset, mDataType = "Kallisto_0.46.1.rnaseq")) 
  gene_ann <- gene_ann[colnames(exp_mat),] # Removing "PAR_Y" genes annotations
  
  # Stripping off the version number from ENSEMBLE IDs
  # colnames(exp_mat) <- sub("\\..*", "", colnames(exp_mat))
  # rownames(gene_ann) <- sub("\\..*", "", rownames(gene_ann))
  
  cell_ann_seq <- cellInfo(pset) 
  
  exp_mat <- exp_mat[intersect(rownames(cell_ann_seq ) , rownames(exp_mat)), ] 
  cell_ann_seq <- cell_ann_seq[intersect(rownames(cell_ann_seq), rownames(exp_mat)), ]
  
  pgx.dat <- list(rna_seq_exp = exp_mat, rna_seq_gene_ann = gene_ann, 
                  cell_ann_seq = cell_ann_seq)
  
  return(pgx.dat)
  
}

## Drug Response PSets

extractDrugData  <- function(pset){
  
  drug_aac <- summarizeSensitivityProfiles(pset, 
                                           sensitivity.measure = "aac_recomputed")
  
  cell_ann_seq <- cellInfo(pset) 
  drug_aac <- drug_aac[, intersect(rownames(cell_ann_seq ) , colnames(drug_aac))] 
  cell_ann_seq <- cell_ann_seq[intersect(rownames(cell_ann_seq), colnames(drug_aac)), ]
  
  pgx.dat <- list( cell_ann_seq = cell_ann_seq, drug_aac = drug_aac)
  
  return(pgx.dat)
  
}

#################################
## CCLE PSets
#################################

dir <- "C:/PGx_SARC"
dat <- readRDS(file.path(dir, "CCLE.rds"))

# add this line to update the downloaded object from ORCESTRA
dat <- updateObject(dat)

#################################################################
# extract the rna-seq, gene annotation, and cell info/annotation
#################################################################
dat_ccle <- extractRNASeqData(dat)

class(dat_ccle)

names(dat_ccle)

# rna-seq: log(TPM+0.001)
dat_expr <- dat_ccle$rna_seq_exp #60617 genes and 1019 cell lines

# gene annotation
dat_annot <- dat_ccle$rna_seq_gene_ann

# cell information/annotation
dat_cell <- dat_ccle$cell_ann_seq

#####################################
# extract the drug response data
#####################################

dat_drug <- extractDrugData(dat)

class(dat_drug)

names(dat_drug)

## aac matrix
dat_acc <- dat_drug$drug_aac # 24 drugs and 1094 cell lines

## cell annotation/information (it is the same as line 78)
dat_cell <- dat_drug$cell_ann_seq