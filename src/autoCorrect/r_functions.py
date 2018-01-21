import readline
from rpy2.robjects.packages import importr
fitdistrplus = importr("fitdistrplus")
base = importr('base')
import rpy2.robjects as robjects
r = robjects.r
import numpy as np

size_factors = np.array(robjects.r('''
    library(DESeq2)
    get_sf <- function(data, names, verbose=FALSE) {
        data <-t(data)
        matrix <-data.matrix(data, rownames.force=F)
        sample <- factor(names)
        desq_data <-DESeqDataSetFromMatrix(matrix, DataFrame(sample), design= ~ sample)
        ddq <- estimateSizeFactors(desq_data)
        sf <-sizeFactors(ddq)
        return(sf)
    }
'''))