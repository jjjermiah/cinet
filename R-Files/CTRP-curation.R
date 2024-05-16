## load libraries
library(qs)
library(dplyr)
library(PharmacoGx)

dir <- "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/PSets"
dat <- readRDS(file.path(dir, "PSet_CTRPv2.rds"))
# add this line to update the downloaded object from ORCESTRA
dat <- updateObject(dat)

dat_cells <- dat@sample
dat_drugs <- dat@treatment
dat_experiments <- cbind(dat@treatmentResponse$info, dat@treatmentResponse$profiles)

write.csv(dat_cells, "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/PSets/csv/CTRP-cells.csv", row.names=TRUE)
write.csv(dat_drugs, "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/PSets/csv/CTRP-drugs.csv", row.names=TRUE)
write.csv(dat_experiments, "C:/Users/marcd/OneDrive/Escritorio/UHN/DeepCINET/PSets/csv/CTRP-exps.csv", row.names=TRUE)
