install.packages("qs")

library(qs)

## Read the qs data

dat <- qread("C:/Users/marcd/Downloads/GSE15622_SE.qs")
## head(dat@colData@listData)

## save as csv file

write.csv(dat@colData@listData, file ="C:/Users/marcd/Downloads/GSE15622_SE.csv")
write.csv(dat@assays@data@listData, file ="C:/Users/marcd/Downloads/GSE15622_SE_expr.csv")
