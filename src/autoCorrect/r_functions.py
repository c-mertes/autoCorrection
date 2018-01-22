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

def get_size_factor(self):
        get_sf = robjects.r['get_sf']
        self.sample_names = r.c(self.sample_names)
        sf = np.array(get_sf(self.data, self.sample_names))
        sf[sf == 0] = 1
        sf = sf.reshape(sf.shape[0],1)
        return sf