from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np
import scipy as sp
import pandas as pd
import random
import h5py
import dask.array as da
#from rpy2.robjects.packages import importr
#from rpy2.robjects.vectors import FloatVector
##
#import readline
#from rpy2.robjects.packages import importr
#base = importr('base')
#import rpy2.robjects as robjects
#r = robjects.r
#import rpy2.robjects.numpy2ri
#rpy2.robjects.numpy2ri.activate()
#from .r_functions import *
from collections import OrderedDict
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from ggplot import *
from copy import deepcopy


class Simulation():
    def __init__(self, num_sample=1000, num_feat=2,
                 num_genes=10000, dispersion_min=25,
                 dispersion_max=25, adjusted_median=1000,
                 counts_file="counts"):

        self.num_sample = num_sample
        self.num_feat = num_feat
        self.num_genes = num_genes
        self.dispersion_min = dispersion_min
        self.dispersion_max = dispersion_max
        self.latent_var = self.get_latent_var()
        self.log_means = self.get_log_means()
        self.not_scaled_means = self.get_not_scaled_means()
        self.dispersion = self.get_dispersion()
        self.adjusted_median = adjusted_median
        self.means = self.get_means()
        self.counts = self.get_counts()
        self.counts_file = counts_file

    def get_random_nbinom(self, n=None):
        #eps = 1e-10
        # translate these into gamma parameters
        gamma_shape = self.dispersion
        gamma_scale = self.means / gamma_shape
        gamma_samples = np.random.gamma(gamma_shape, gamma_scale, n)
        return np.random.poisson(gamma_samples)

    def get_latent_var(self):
        self.latent_var =  np.random.normal(0, 0.5, (self.num_sample, self.num_feat)).astype(np.float32)
        return self.latent_var

    def get_log_means(self):
        np.random.seed(18)
        z = self.latent_var
        W = np.random.normal(0, 0.5, (self.num_feat, self.num_genes)).astype(np.float32)
        b = np.random.normal(0, 0.5, (1, self.num_genes)).astype(np.float32)
        self.log_means = np.dot(z, W) + b
        return self.log_means

    def get_not_scaled_means(self):
        self.not_scaled_means = np.exp(self.log_means)
        return self.not_scaled_means

    def get_dispersion(self):
        dispersion = np.random.uniform(self.dispersion_min,
                                       self.dispersion_max, [1, self.num_genes])
        self.dispersion = np.zeros_like(self.not_scaled_means) + dispersion
        return self.dispersion

    def get_means(self):
        scale = np.multiply(self.adjusted_median, self.not_scaled_means) #adjust median
        mean_of_medians = np.mean(np.median(self.not_scaled_means,axis=0))
        self.means = scale / mean_of_medians
        return self.means

    def get_counts(self):
        self.counts = self.get_random_nbinom()
        return self.counts.astype(int)
    
    def write_in_h5(self):
        counts = da.from_array(self.counts.astype(int),
                          chunks=(self.counts.shape[0],
                                  self.counts.shape[1]))
        means = da.from_array(self.means,
                          chunks=(self.means.shape[0],
                                  self.means.shape[1]))
        z = da.from_array(self.latent_var,
                          chunks=(self.latent_var.shape[0],
                                  self.latent_var.shape[1]))
        da.to_hdf5(self.counts_file+'.h5', {'/counts': counts,
                                   '/means': means,'/z': z})

    def write_counts_in_h5(self):
        with h5py.File(self.counts_file+".h5", 'w') as outfile:
            outfile.create_dataset('simulated_counts', data=self.counts.astype(int))

    def write_counts_in_csv(self):
        counts = pd.DataFrame(self.counts.astype(int))
        counts.to_csv(self.counts_file+".csv", index=True, header=True, sep='\t')
        #np.savetxt(self.counts_file+".csv", self.counts.astype(int), delimiter=',')


class OutlierData():
    def __init__(self, index, data_with_outliers):
        self.index = index
        self.data_with_outliers = data_with_outliers
  
        
class OutlierInjection():
    def __init__(self, input_data, inject_zeros=False,
                 inject_plus_minus_one=False,
                 nr_of_smples_with_outliers=None,
                 min_samples_with_outliers=200,
                 max_samples_with_outliers=200,
                 min_genes_with_outliers=900,
                 max_genes_with_outliers=1000,
                 counts_file="outlier_counts",
                 inj_lower_expr=False, fold=5,
                 sample_names=None, gene_names=None):
        
        self.inj_lower_expr=inj_lower_expr
        if fold is None:
            self.fold=self.set_fold()
        else:
            self.fold=fold
        self.input_data = input_data
        self.log2fc = self.computeLog2foldChange()
        self.inject_zeros = inject_zeros
        self.inject_plus_minus_one = inject_plus_minus_one
        self.min_samples_with_outliers = min_samples_with_outliers
        self.max_samples_with_outliers = max_samples_with_outliers + 1       
        self.min_genes_with_outliers = min_genes_with_outliers
        self.max_genes_with_outliers = max_genes_with_outliers + 1
        self.outlier_data = self.get_outlier_data()
        self.counts_file = counts_file
        self.sample_names = sample_names
        self.gene_names = gene_names 
        
    def computeLog2foldChange(self):
        log2fc = np.log2((0.00001 + self.input_data)/(0.00001 + np.mean(self.input_data, axis=0)))
        return log2fc
    
    def set_fold(self):
        if self.inj_lower_expr:
            fold = np.trunc(np.min(self.log2fc))
        else:
            fold = np.trunc(np.max(self.log2fc))
        return fold

    def get_outlier_data(self):
        my_data_with_outliers = np.copy(self.input_data)
        indexesOfOutliers = np.zeros_like(self.input_data)
        nr_of_smples_with_outliers = np.random.choice(np.arange(self.min_samples_with_outliers,
                                                      self.max_samples_with_outliers))
        indexOfOutlierSample = random.sample(range(self.input_data.shape[0]),
                                             nr_of_smples_with_outliers)
        for i in indexOfOutlierSample:
            nrOfOutliers = np.random.choice(np.arange(self.min_genes_with_outliers,
                                                      self.max_genes_with_outliers))
            indexOfOutlierGene = random.sample(range(self.input_data.shape[1]),
                                               nrOfOutliers)
            for j in indexOfOutlierGene:
                if self.inject_zeros:
                    if self.log2fc[i][j] > 4:
                        indexesOfOutliers[i][j] = 1
                        my_data_with_outliers[i][j] = 0
                elif self.inject_plus_minus_one:
                    indexesOfOutliers[i][j] = 1
                    deltaX = np.random.choice((-1,1))
                    my_data_with_outliers[i][j] = self.input_data[i][j]+deltaX
                else:
                    indexesOfOutliers[i][j] = 1
                    my_data_with_outliers[i][j] = np.round(np.mean(self.input_data[:,j])*np.power(2.0,self.fold)) 
        return OutlierData(indexesOfOutliers, my_data_with_outliers)

    def write_in_h5(self):
        x = da.from_array(self.outlier_data.data_with_outliers.astype(int),
                          chunks=(self.outlier_data.data_with_outliers.shape[0],
                                  self.outlier_data.data_with_outliers.shape[1]))
        y = da.from_array(self.outlier_data.index.astype(int),
                          chunks=(self.outlier_data.index.shape[0],
                                  self.outlier_data.index.shape[1]))
        da.to_hdf5(self.counts_file+'.h5', {'/counts': x,
                                   '/idx': y})
        
    def write_in_csv_with_names(self):
        counts = pd.DataFrame(self.outlier_data.data_with_outliers.astype(int),
                          index=self.sample_names, columns=self.gene_names)
        counts.to_csv(self.counts_file+".csv", index=True, header=True, sep='\t')
        idx = pd.DataFrame(self.outlier_data.index.astype(int),
                          index=self.sample_names, columns=self.gene_names)
        idx.to_csv(self.counts_file+"_outlier_idx.csv", index=True, header=True, sep='\t')
 
                     
    def write_counts_in_csv(self):
        np.savetxt(self.counts_file+".csv", self.outlier_data.data_with_outliers.astype(int), delimiter=',')

                    
class OutInjection(OutlierInjection):
    def __init__(self, input_data, outlier_prob=10.0**-4, fold=5, inj_lower_expr=False,
                 sample_names=None, gene_names=None, counts_file = "out_file"):
        
        self.input_data=input_data
        self.outlier_prob=outlier_prob
        if fold is None:
            self.log2fc = self.computeLog2foldChange()
            self.fold=self.set_fold()
        else:
            self.fold=fold
        self.outlier_data = self.get_outlier_data()
        self.sample_names = sample_names
        self.gene_names = gene_names
        self.counts_file = counts_file
            
    def get_outlier_data(self):
        injected = np.copy(self.input_data)
        data = self.input_data.flatten()
        idx=np.random.choice((-1,1,0), size=(np.multiply(self.input_data.shape[0],
                                                         self.input_data.shape[1])),
                             p=(self.outlier_prob/2, self.outlier_prob/2, 1-self.outlier_prob))
        places = np.array(range(0, data.shape[0]))
        for entry, indicator, place in zip(data, idx, places):
            if indicator == -1:
                if self.fold > 0: self.fold = indicator * self.fold
                i = np.unravel_index(place, self.input_data.shape)[0]
                j = np.unravel_index(place, self.input_data.shape)[1]
                injected[i][j] = round(np.mean(self.input_data[:,j])*(2.0**self.fold))
            elif indicator == 1:
                if self.fold < 0: self.fold = abs(self.fold)
                i = np.unravel_index(place, self.input_data.shape)[0]
                j = np.unravel_index(place, self.input_data.shape)[1]
                injected[i][j] = round(np.mean(self.input_data[:,j])*(2.0**self.fold))
        idx = idx.reshape(self.input_data.shape[0],self.input_data.shape[1])
        return OutlierData(idx, injected)


class OutInjectionZscoreFC(OutlierInjection):
    def __init__(self, input_data, outlier_prob=10.0**-3,
                 use_z_score=True, z=4, fold=None, inj_lower_expr=False,
                 sample_names=None, gene_names=None, counts_file = "out_file"):
        
        self.input_data=input_data
        self.outlier_prob=outlier_prob
        if use_z_score:
            self.z = self.set_score()
            self.log2fc = self.computeLog2foldChange()
            self.fold = self.set_fold_z()
        else:
            if fold is None:
                self.fold = self.set_score()
            else:
                self.fold = np.full((1, input_data.shape[1]), fold, dtype=int)
        self.outlier_data = self.get_outlier_data()
        self.sample_names = sample_names
        self.gene_names = gene_names
        self.counts_file = counts_file
        print("doing!")
            
    def set_fold_z(self):
        fold = self.z * np.std(self.log2fc, axis=0) + np.mean(self.log2fc, axis=0)
        return fold
    
    def set_score(self):
        score = np.random.choice((3,4,5), self.input_data.shape[1])
        return score
            
    def get_outlier_data(self):
        injected = np.copy(self.input_data)
        data = self.input_data.flatten()
        idx=np.random.choice((-1,1,0), size=(np.multiply(self.input_data.shape[0],
                                                         self.input_data.shape[1])),
                             p=(self.outlier_prob/2, self.outlier_prob/2, 1-self.outlier_prob))
        places = np.array(range(0, data.shape[0]))
        for entry, indicator, place in zip(data, idx, places):
            if indicator == -1:
                i = np.unravel_index(place, self.input_data.shape)[0]
                j = np.unravel_index(place, self.input_data.shape)[1]
                if self.fold[j] > 0: self.fold[j] = indicator * self.fold[j]
                out_count = round(np.mean(self.input_data[:,j])*(2.0**self.fold[j]))
                injected[i][j] = out_count
            elif indicator == 1:
                i = np.unravel_index(place, self.input_data.shape)[0]
                j = np.unravel_index(place, self.input_data.shape)[1]
                if self.fold[j] < 0: self.fold[j] = abs(self.fold[j])
                out_count = round(np.mean(self.input_data[:,j])*(2.0**self.fold[j]))
                if out_count > 200000: #100*np.max(self.input_data):
                    injected[i][j] = 200000
                else:
                    injected[i][j] = out_count
        idx = idx.reshape(self.input_data.shape[0],self.input_data.shape[1])
        return OutlierData(idx, injected)

    
class OutInjectionFC(OutlierInjection):
    def __init__(self, input_data, outlier_prob=10.0**-3,
                 fold=None, sample_names=None, gene_names=None,
                 counts_file = "out_file"):
        
        self.input_data=input_data
        self.outlier_prob=outlier_prob
        if fold is None:
            self.log2fc = self.computeLog2foldChange()
            self.fold = self.set_fold()
        else:
            self.fold = np.full((input_data.shape[1]), fold, dtype=int)
        self.outlier_data = self.get_outlier_data()
        self.sample_names = sample_names
        self.gene_names = gene_names
        self.counts_file = counts_file
        print("Injecting!")
            
    def set_fold(self):
        fc_mins = np.trunc(np.min(self.log2fc, axis=0))
        fc_maxs = np.trunc(np.max(self.log2fc, axis=0))
        fold = np.stack([fc_mins, fc_maxs])        
        return fold
            
    def get_outlier_data(self):
        injected = np.copy(self.input_data)
        data = self.input_data.flatten()
        idx=np.random.choice((1,0), size=(np.multiply(self.input_data.shape[0], 
                                                      self.input_data.shape[1])),
                                                      p=(self.outlier_prob,
                                                      1-self.outlier_prob))
        places = np.array(range(0, data.shape[0]))
        for entry, indicator, place in zip(data, idx, places):
            if indicator == 1:
                i = np.unravel_index(place, self.input_data.shape)[0]
                j = np.unravel_index(place, self.input_data.shape)[1]
                if self.log2fc[i][j] >= 0:
                    fold = self.fold[0][j]-1
                else:
                    fold = self.fold[1][j]+1
                out_count = round(np.mean(self.input_data[:,j])*(2.0**fold))
                if out_count > 200000: #100*np.max(self.input_data):
                    injected[i][j] = 200000
                else:
                    injected[i][j] = out_count
        idx = idx.reshape(self.input_data.shape[0],self.input_data.shape[1])
        return OutlierData(idx, injected)

    
class OutInjectionFCfixed(OutlierInjection):
    def __init__(self, input_data, outlier_prob=10.0**-3,
                 fold=None, sample_names=None, gene_names=None,
                 counts_file = "out_file"):
        
        self.input_data=input_data
        self.outlier_prob=outlier_prob
        self.log2fc = self.computeLog2foldChange()
        self.outlier_data = self.get_outlier_data()
        self.sample_names = sample_names
        self.gene_names = gene_names
        self.counts_file = counts_file
        print("Injecting!")

    def get_outlier_data(self):
        injected = np.copy(self.input_data)
        data = self.input_data.flatten()
        idx=np.random.choice((1,0), size=(np.multiply(self.input_data.shape[0], 
                                                      self.input_data.shape[1])),
                                                      p=(self.outlier_prob,
                                                      1-self.outlier_prob))
        places = np.array(range(0, data.shape[0]))
        for entry, indicator, place in zip(data, idx, places):
            if indicator == 1:
                i = np.unravel_index(place, self.input_data.shape)[0]
                j = np.unravel_index(place, self.input_data.shape)[1]
                if self.log2fc[i][j] >= 0:
                    out_count = 0
                else:
                    out_count = 200000
                injected[i][j] = out_count
        idx = idx.reshape(self.input_data.shape[0],self.input_data.shape[1])
        return OutlierData(idx, injected)

    
class OutInjectionFCzero(OutlierInjection):
    def __init__(self, input_data, outlier_prob=10.0**-3,
                 fold=None, sample_names=None, gene_names=None,
                 counts_file = "out_file"):
        
        self.input_data=input_data
        self.outlier_prob=outlier_prob
        self.log2fc = self.computeLog2foldChange()
        self.outlier_data = self.get_outlier_data()
        self.sample_names = sample_names
        self.gene_names = gene_names
        self.counts_file = counts_file
        print("injecting zeros!")

    def get_outlier_data(self):
        injected = np.copy(self.input_data)
        data = self.input_data.flatten()
        idx=np.random.choice((1,0), size=(np.multiply(self.input_data.shape[0], 
                                                      self.input_data.shape[1])),
                                                      p=(self.outlier_prob,
                                                      1-self.outlier_prob))
        places = np.array(range(0, data.shape[0]))
        for entry, indicator, place in zip(data, idx, places):
            if indicator == 1:
                i = np.unravel_index(place, self.input_data.shape)[0]
                j = np.unravel_index(place, self.input_data.shape)[1]
                if self.log2fc[i][j] >= 0:
                    injected[i][j] = 0
        idx = idx.reshape(self.input_data.shape[0],self.input_data.shape[1])
        return OutlierData(idx, injected)

    
class ZeroInjection(OutlierInjection):
    def __init__(self, input_data, outlier_prob=10.0**-4,
                 sample_names=None, gene_names=None, counts_file = "out_file"):
        
        self.input_data=input_data
        self.outlier_prob=outlier_prob
        self.outlier_data = self.get_outlier_data()
        self.sample_names = sample_names
        self.gene_names = gene_names
        self.counts_file = counts_file
            
    def get_outlier_data(self):
        injected = np.copy(self.input_data)
        data = self.input_data.flatten()
        idx=np.random.choice((1,0), size=(np.multiply(self.input_data.shape[0],
                                                         self.input_data.shape[1])),
                             p=(self.outlier_prob, 1-self.outlier_prob))
        places = np.array(range(0, data.shape[0]))
        for entry, indicator, place in zip(data, idx, places):
            if indicator == 1:
                i = np.unravel_index(place, self.input_data.shape)[0]
                j = np.unravel_index(place, self.input_data.shape)[1]
                injected[i][j] = 0
        idx = idx.reshape(self.input_data.shape[0],self.input_data.shape[1])
        return OutlierData(idx, injected)

    
class ZeroInjectionFixed(OutlierInjection):
    def __init__(self, input_data, from_fc=2, to_fc=4, sample_names=None,
                 gene_names=None, counts_file = "out_file"):
        self.input_data=input_data
        self.log2fc = self.computeLog2foldChange()
        self.from_fc=from_fc
        self.to_fc=to_fc
        self.outlier_data = self.get_outlier_data()
        self.sample_names = sample_names
        self.gene_names = gene_names
        self.counts_file = counts_file
            
    def get_outlier_data(self):
        ii = np.where((self.log2fc>self.from_fc) & (self.log2fc<self.to_fc))
        indexes_of_ii = np.unique(ii[0], return_index=True)[1]
        a = [ii[0][index] for index in sorted(indexes_of_ii)]
        b = [ii[1][index] for index in sorted(indexes_of_ii)]
        id_for_idx =(np.asarray(a),np.asarray(b)) 
        idx =np.zeros((self.input_data.shape[0], self.input_data.shape[1]))
        idx[id_for_idx] = 1
        injected = np.copy(self.input_data)
        injected[id_for_idx] = 0
        return OutlierData(idx, injected)
    
    
class ZeroInjectionWhereMean(OutlierInjection):
    def __init__(self, input_data, from_fc=1, min_gene_mean=1000,
                 sample_names=None, frac_of_out = 10.0**-4,
                 gene_names=None, counts_file = "out_file"):
        self.input_data=input_data
        self.log2fc = self.computeLog2foldChange()
        self.from_fc=from_fc
        self.frac_of_out = frac_of_out 
        self.min_gene_mean = min_gene_mean
        self.ind_high_mean_and_fc = self.get_high_mean_and_fc_idx()
        self.outlier_data = self.get_outlier_data()
        self.sample_names = sample_names
        self.gene_names = gene_names
        self.counts_file = counts_file
        
    def get_high_mean_and_fc_idx(self):
        self.high_fc = np.where((self.log2fc>self.from_fc))
        high_mean = np.where(np.mean(self.input_data, axis=0)>self.min_gene_mean)
        ind_high_mean_and_fc = np.nonzero(np.in1d(self.high_fc[1], high_mean[0]))[0]
        #if ind_high_mean_and_fc.size == 0: Except("Set lower from_fc, or lower min_gene_mean")
        return ind_high_mean_and_fc
    
    def get_nr_of_outliers(self):
        nr = round(self.input_data.shape[0]*self.input_data.shape[1]*self.frac_of_out)
        return nr
    
    def set_nr_of_outliers(self):
        nr = self.get_nr_of_outliers()
        if self.ind_high_mean_and_fc.size > nr:
            nr_of_out = nr
        else:
            nr_of_out = self.ind_high_mean_and_fc.size
        return nr_of_out
            
    def get_outlier_data(self):
        self.nr_of_out = self.set_nr_of_outliers()
        ind_high_mean_and_fc = np.random.choice(self.ind_high_mean_and_fc,
                                                self.nr_of_out, replace=False)
        i=self.high_fc[0][ind_high_mean_and_fc]
        j=self.high_fc[1][ind_high_mean_and_fc]
        idx =np.zeros((self.input_data.shape[0],
                       self.input_data.shape[1]))
        idx[(i,j)] = 1
        injected = np.copy(self.input_data)
        injected[(i,j)] = 0
        return OutlierData(idx, injected)

    
class FoldChInjectionWhereMean(OutlierInjection): # <----change to fraction
    def __init__(self, input_data, fc_treshold=1, gene_mean_treshold=1000,
                 sample_names=None, nr_of_out = 800, fold=-5, use_large_out_val=False,
                 gene_names=None, counts_file = "out_file", use_log2fc_out=True):
        self.input_data = input_data
        self.use_log2fc_out = use_log2fc_out
        self.log2fc = self.computeLog2foldChange()
        self.fc_treshold=fc_treshold
        self.fold = fold
        self.nr_of_out = nr_of_out 
        self.gene_mean_treshold = gene_mean_treshold
        self.use_large_out_val = use_large_out_val
        self.outlier_data = self.get_outlier_data()
        self.sample_names = sample_names
        self.gene_names = gene_names
        self.counts_file = counts_file
            
    def get_outlier_data(self):
        if self.use_large_out_val:
            fc_id = np.where((self.log2fc<self.fc_treshold))
            mean_id = np.where(np.mean(self.input_data, axis=0)<self.gene_mean_treshold)
        else:
            fc_id = np.where((self.log2fc>self.fc_treshold))
            mean_id = np.where(np.mean(self.input_data, axis=0)>self.gene_mean_treshold)
            
        ind_mean_and_fc = np.nonzero(np.in1d(fc_id[1], mean_id[0]))[0]
        ind_mean_and_fc = np.random.choice(ind_mean_and_fc, self.nr_of_out, replace=False)
        i=fc_id[0][ind_mean_and_fc]
        j=fc_id[1][ind_mean_and_fc]
        idx =np.zeros((self.input_data.shape[0],
                       self.input_data.shape[1]))
        idx[(i,j)] = 1
        injected = np.copy(self.input_data)
        if self.use_log2fc_out:
            injected[(i,j)] = round(np.mean(self.input_data[:,j])*(2.0**self.fold))
        else:
            injected[(i,j)] = round(np.mean(self.input_data[:,j])*(abs(self.fold)))
        return OutlierData(idx, injected)
   
    
class InjectionOfOutliersDuringSimulation():
    def __init__(self, simulated_data=Simulation(),
                 injection_place="log-means", inject_zeros=False):
        self.simulated_data = simulated_data
        self.injection_place = injection_place
        self.inject_zeros = inject_zeros
        self.injection_data = self.set_injection_data()
        self.outlier_data = self.get_outlier_data_obj()
        self.data_with_outliers = self.get_data_with_outliers()

    def set_injection_data(self):
        if self.injection_place == "log-means":
            self.injection_data = self.simulated_data.get_log_means()
        elif self.injection_place == "means":
            self.injection_data = self.simulated_data.get_means()
        else:
            raise ValueError('Please define "injection_place=" argument as "log-means"(default) or "means"')
        return self.injection_data

    def get_outlier_data_obj(self):
        outlier_injection = OutlierInjection(self.injection_data, inject_zeros=False,
                                             inject_plus_minus_one=True)
        self.outlier_data = outlier_injection.get_outlier_data()
        return self.outlier_data

    def get_data_with_outliers(self):
        outliers = self.outlier_data.data_with_outliers
        if self.injection_place == "log-means":
            self.simulated_data.log_means = outliers
            self.simulated_data.get_not_scaled_means()
            self.simulated_data.get_means()
        elif self.injection_place == "means":
            self.simulated_data.means = outliers
        self.simulated_data.get_counts()
        return self.simulated_data

    
class ProcessedData():
    def __init__(self, x, sf=None):
        self.data = x
        self.size_factor = sf
        
    
class TrainTestData():
    def __init__(self, x_train, x_test, sf_train=None, sf_test=None):
        self.train = x_train
        self.test = x_test
        self.size_factor_train = sf_train
        self.size_factor_test = sf_test

        
class TrainTestPreparation():
    def __init__(self, data, sf=None,
                 sample_names=None,
                 rescale_per_gene=False,
                 rescale_per_sample=False,
                 rescale_by_global_median=True,
                 divide_by_sf=False,
                 no_rescaling=True, ones_sf=False,
                 no_splitting=False):
        self.data = data
        self.sf = sf
        self.sample_names = sample_names
        self.rescale_per_gene = rescale_per_gene
        self.rescale_per_sample = rescale_per_sample
        self.rescale_by_global_median = rescale_by_global_median
        self.ones_sf = ones_sf
        self.data = self.clip_high_values()
        self.set_sf()
        if no_rescaling:
            self.splited_data = self.split_data(self.sf)
        else:
            if divide_by_sf:
                self.data = self.get_rescaled_by_sf()
                self.scaling_factor = self.sf
            else:
                self.scaling_factor = self.get_scaling_factor()
            if no_splitting:
                self.processed_data = self.get_processed_data()
            else:
                self.splited_data = self.split_data(self.scaling_factor)
            
    def clip_high_values(self):
        cliped_data = deepcopy(self.data)
        cliped_data[cliped_data > 200000] = 200000
        return cliped_data

    def get_median_factor(self, data, axis=None):      
        if axis is None: # use global median 
            median_factor = np.median(data)
            median_factor = np.repeat(median_factor, data.shape[1])
        elif axis==0: #factor is median per gene
            median_factor = np.median(data, axis)
            median_factor[median_factor == 0] = 1
        elif axis==1:
            median_factor = np.median(data, axis)
            median_factor[median_factor == 0] = 1
        return median_factor
    
    def get_size_factor(self): #  <--------------------- change this function later
        ##get_sf = robjects.r['get_sf']
        ##self.sample_names = r.c(self.sample_names)
        ##sf = np.array(get_sf(self.data, self.sample_names))
        ##sf[sf == 0] = 1
        ##sf = sf.reshape(sf.shape[0],1)
        self.sf = np.ones_like(self.data)
        return sf
    
    def set_sf(self):
        if self.sf is None:
            if self.ones_sf:
                self.sf = np.ones_like(self.data)
            else:
                self.sf = self.get_size_factor()
    
    def get_scaling_factor(self):        
        if self.rescale_per_gene:
            median_factor = self.get_median_factor(self.data, axis=0)
        elif self.rescale_per_sample:
            median_factor = self.get_median_factor(self.data, axis=1)
            median_factor = median_factor.reshape(median_factor.shape[1], median_factor.shape[0])
            median_factor = np.repeat(median_factor, self.data.shpe[1], axis=1)
        elif self.rescale_by_global_median:
            median_factor = self.get_median_factor(self.data)
        scaling_factor = np.multiply(self.sf, median_factor)
        scaling_factor = np.power(scaling_factor, -1.0)
        return scaling_factor
    
    def get_rescaled_by_sf(self):
        self.data = self.data/self.sf
        return self.data

    def split_data(self, factor):
        x_train, x_test = train_test_split(self.data,
                                           random_state=False,
                                           test_size=0.1)
        sf_train, sf_test = train_test_split(factor,
                                       random_state=False,
                                       test_size=0.1)
        self.splited_data = TrainTestData(x_train, x_test,
                                          sf_train, sf_test)
        return self.splited_data
    
    def get_processed_data(self):
        self.processed_data = ProcessedData(self.data, self.scaling_factor)
        return self.processed_data
    
        
class TrainTestPreparationWithRescaling():
    def __init__(self, data,
                 rescale_per_gene=False,
                 use_rescaling=True):
        self.data = data
        self.rescale_per_gene = rescale_per_gene
        if use_rescaling:
            self.splited_data = self.split_data()
        else:
            self.splited_data = self.split_data_no_rescaling()

    def rescale_data(self, data, x_train, x_test, axis):
        size_factor = np.median(data, axis)# <----------------- maybe mean?
        size_factor[size_factor == 0] = 1
        
        if axis == 1:#size factor is median of each row 
            #(rescaling each sample to have lower values, but same distribution)
            sf_train, sf_test = train_test_split(size_factor, random_state=False,
                                             test_size=0.1)
            sf_train = sf_train[:,None]
            sf_test = sf_test[:,None]
        else: #(rescaling each gene to have lower values, but same distribution)
            sf_train = size_factor
            sf_test = size_factor

        x_train_rescaled = x_train.astype('float32') / sf_train
        x_test_rescaled = x_test.astype('float32') / sf_test
        return TrainTestData(x_train_rescaled, x_test_rescaled)

    def split_data(self):
        x_train, x_test = train_test_split(self.data,
                                           random_state=False,
                                           test_size=0.1)
        if self.rescale_per_gene:
            self.splited_data = self.rescale_data(self.data, x_train, x_test, axis=0)
        else:
            self.splited_data = self.rescale_data(self.data, x_train, x_test, axis=1)
        return self.splited_data

    def split_data_no_rescaling(self):
        x_train, x_test = train_test_split(self.data, random_state=False,
                                           test_size=0.1)
        self.splited_data = TrainTestData(x_train, x_test)
        return self.splited_data    
    
    
class RescaleBack():
    def __init__(self, orig_data, data_scaled,
                 rescale_per_gene=False):
        self.orig_data = orig_data
        self.data_scaled = data_scaled
        if rescale_per_gene:
            self.axis = 0
        else:
            self.axis = 1
        self.resc_back_data = self.get_rescaled_back()
    
    def get_rescaled_back(self):
        size_factor = np.median(self.orig_data, self.axis)
        if self.axis == 1: #multiply each col
            self.resc_back_data = self.data_scaled * size_factor[:, np.newaxis]
        elif self.axis == 0: #mult each row
            self.resc_back_data = self.data_scaled * size_factor
        return self.resc_back_data

    
class DataReader():
    def __init__(self):
        pass
    
    def read_gtex_skin(self):
        path="/s/project/scared/GTEx/filtered_counts18k.tsv"
        self.data = self.read_data(path, sep=" ")
        return self.data
    
    def read_data(self, path, sep):
        data_pd = pd.read_csv(path, index_col=0,header=0, sep=sep)
        data = np.transpose(np.array(data_pd.values))
        return data
    
    
class DataCooker():
    def __init__(self, counts, size_factors=None,
                 inject_outliers=True, inj_method="OutInjectionZscoreFC",
                 pred_counts=None, pred_sf=None):
        self.counts=counts
        self.inject_outliers=inject_outliers
        self.inj_method = inj_method
        if size_factors is not None:
            self.sf = size_factors
        else:
            self.sf = np.ones_like(counts).astype(float)
        if pred_counts is not None:
            self.pred_counts = pred_counts
            if pred_sf is not None:
                self.pred_sf = pred_sf
            else:
                self.pred_sf = np.ones_like(counts).astype(float)
        else:
            self.pred_counts = deepcopy(counts)
            self.pred_sf = self.sf
      
    def inject(self, data):
        print("Using "+self.inj_method+" method!")
        if self.inj_method == "zeroWhereMean":
            injected_outliers = ZeroInjectionWhereMean(data)
        elif self.inj_method == "OutInjectionFC":
            injected_outliers = OutInjectionFC(data)
        elif self.inj_method == "OutInjectionZscoreFC":
            injected_outliers = OutInjectionZscoreFC(data)
        elif self.inj_method == "OutInjectionFCzero":
            injected_outliers = OutInjectionFCzero(data)
        elif self.inj_method == "OutInjectionFCfixed":
            injected_outliers = OutInjectionFCfixed(data)
        else:
            raise ValueError("Please specify one of injection methods: 'zeroWhereMean', 'fcWhereMean', 'fcRandom', 'fcZscoreRandom'")
        return injected_outliers
    
    def get_count_data(self, counts, sf):
        count_data = TrainTestPreparation(data=counts,sf=sf,
                                  no_rescaling=False,
                                  no_splitting=True)
        return count_data
    
    def prepare_rescaled(self, count_data, sf):
        rescaled = TrainTestPreparation(
                     data=count_data.processed_data.data, sf=sf,
                     no_rescaling=False, no_splitting=False,
                     divide_by_sf=True)
        return rescaled
    
    def prepare_simple(self, count_data):
        rescaled_simple = TrainTestPreparation(
                     data=count_data.processed_data.data,
                     sf=count_data.processed_data.size_factor,
                     no_rescaling=True, no_splitting=False)
        return rescaled_simple
    
    def prepare_noisy(self, count_data):
        if self.inject_outliers:
            inj = self.inject(count_data.processed_data.data)
            noisy_train_test = TrainTestPreparation(
                             data=inj.outlier_data.data_with_outliers,
                             sf=count_data.processed_data.size_factor,
                             no_rescaling=True, no_splitting=False)
        else:
            noisy_train_test = self.prepare_simple(count_data)
        return noisy_train_test
    
    def prepare_pred(self, pred_count_data):  
        pred_noisy = self.inject(pred_count_data.processed_data.data)
        return pred_noisy
                           
    def data(self, inj_method="OutInjectionZscoreFC"):
        self.inj_method=inj_method
        count_data = self.get_count_data(self.counts,self.sf)
        pred_count_data = deepcopy(count_data)
        #rescaled = self.prepare_rescaled(count_data, self.sf)
        simple_train_test = self.prepare_simple(count_data)
        noisy_train_test = self.prepare_noisy(count_data) 
        if self.inject_outliers:
            if not np.array_equal(self.counts,self.pred_counts):
                pred_count_data = self.get_count_data(self.pred_counts,self.pred_sf)
            pred_noisy = self.prepare_pred(pred_count_data)
            x_2nd_noise_test = {'inp': pred_noisy.outlier_data.data_with_outliers, 
                                'sf': pred_count_data.processed_data.size_factor}
            y_true_idx_test = np.stack([self.pred_counts.astype(int), pred_noisy.outlier_data.index])
        else:
            x_2nd_noise_test = None
            y_true_idx_test = None
            
        x_noisy_train = {'inp': noisy_train_test.splited_data.train,
                         'sf': noisy_train_test.splited_data.size_factor_train}                
        x_train = simple_train_test.splited_data.train        
        x_noisy_valid = {'inp': noisy_train_test.splited_data.test,
                         'sf': noisy_train_test.splited_data.size_factor_test}        
        x_valid = simple_train_test.splited_data.test
   
        return (x_noisy_train, x_train),(x_noisy_valid, x_valid),(x_2nd_noise_test, y_true_idx_test)


class EvaluationOfOutInjection():
    def __init__(self, out_data, out_idx, orig_data=None,
                 q="None", out_fc="None", data_origin=" ", 
                 pdf_name="ev_inj.pdf", save_to_file=False,
                 in_log_scale=False):
        self.q = q
        self.fc = out_fc
        self.in_log = in_log_scale
        self.data_origin = data_origin
        self.pdf_name = pdf_name
        self.out_data = out_data
        self.out_idx = out_idx
        self.orig_data = orig_data
        self.save_to_file = save_to_file
        self.sample_s = self.get_sample_s()
        self.gene_s = self.get_gene_s()
        self.st = self.summary_table()
        self.sample_with_out_idx = self.get_sample_with_out_idx()
        self.gene_with_out_idx = self.get_gene_with_out_idx()
        if orig_data is not None:
            with PdfPages(self.pdf_name) as self.pdf:
                self.plot_injected_vs_no_out_one_sample()
                self.plot_injected_vs_no_out_one_gene()
                self.plot_outliers_vs_true_vals()
        
    def get_sample_s(self):
        s =np.sum(self.out_idx, axis=1)
        return s
    
    def get_gene_s(self):
        s =np.sum(self.out_idx, axis=0)
        return s
    
    def get_number_of_outliers(self):
        s =np.sum(self.sample_s)
        return s
    
    def get_max_per_sample(self):
        return max(self.sample_s)
    
    def get_max_per_gene(self):
        return max(self.gene_s)
    
    def get_min_per_sample(self):
        return min(self.sample_s)
    
    def get_min_per_gene(self):
        return min(self.gene_s)
    
    def get_mean_per_sample(self):
        return np.mean(self.sample_s)
    
    def get_mean_per_gene(self):
        return np.mean(self.gene_s)
    
    def get_sample_with_out_idx(self):
        return np.where(self.sample_s)[0]
    
    def get_gene_with_out_idx(self):
        return np.where(self.gene_s)[0]
    
    def plot_injected_vs_no_out_one_sample(self, idx=0):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('One sample. Middle layer q = %s\nOutlier fold change fc = %s\n%s'%(self.q,self.fc,self.data_origin), fontsize=14)
        ax.scatter(self.out_data[self.sample_with_out_idx[idx]],
                    self.orig_data[self.sample_with_out_idx[idx]],
                    c=self.out_idx[self.sample_with_out_idx[idx]], label="Outlier")
        ax.set_xlabel("Data with outliers")
        ax.set_ylabel("Data without outliers")
        plt.legend(loc="lower right", markerscale=0.7, scatterpoints=1, fontsize=10)
        ax = plt.gca()
        leg = ax.get_legend()
        leg.legendHandles[0].set_color('yellow')
        if self.in_log:
            plt.yscale('log')
            plt.xscale('log')
        plt.show()
        if self.save_to_file:
            self.pdf.savefig(fig)
        plt.close()
    
    def plot_outliers_vs_true_vals(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Middle layer q = %s\nOutlier fold change fc = %s\n%s'%(self.q,self.fc,self.data_origin), fontsize=14)
        idxx =np.where(self.out_idx.flatten())
        outliers = plt.scatter(range(0,len(idxx[0])),
                self.out_data.flatten()[idxx[0][0:len(idxx[0])]],
                marker='o'
                )
        true_vals = plt.scatter(range(0,len(idxx[0])),
                                self.orig_data.flatten()[idxx[0][0:len(idxx[0])]],
                              marker='o')
        if self.in_log:
            plt.yscale("log")
        plt.legend([outliers, true_vals], 
                   ['Injected outliers', 'True values'],
                   loc=(1.04,0))
        ax.set_ylabel("Counts")
        ax.set_xlabel("All samples with injected outliers in random order")
        plt.show()
        if self.save_to_file:
            self.pdf.savefig(fig)
        plt.close()
        
    def plot_injected_vs_no_out_one_gene(self, idx=0):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('One gene. Middle layer q = %s\nOutlier fold change fc = %s\n%s'%(self.q,self.fc,self.data_origin), fontsize=14)
        ax.scatter(self.out_data[:, self.gene_with_out_idx[idx]],
                    self.orig_data[:, self.gene_with_out_idx[idx]],
                    c=self.out_idx[:, self.gene_with_out_idx[idx]], label="Outlier")
        ax.set_xlabel("Data with outliers")
        ax.set_ylabel("Data without outliers")
        plt.legend(loc="lower right", markerscale=0.7, scatterpoints=1, fontsize=10)
        ax = plt.gca()
        leg = ax.get_legend()
        leg.legendHandles[0].set_color('yellow')
        if self.in_log:
            plt.yscale('log')
            plt.xscale('log')
        plt.show()
        if self.save_to_file:
            self.pdf.savefig(fig)
        plt.close()
        
    def summary_table(self):
        d = {'Total nr of outliers': [self.get_number_of_outliers()],
             'min outliers in sample': [self.get_min_per_sample()],
             'max outliers in sample': [self.get_max_per_sample()],
             'mean outliers in sample': [self.get_mean_per_sample()],             
             'min outliers in gene': [self.get_min_per_gene()],
             'max outliers in gene': [self.get_max_per_gene()],
             'mean outliers in gene': [self.get_mean_per_gene()],
            }
        d=OrderedDict(d)
        df = pd.DataFrame(data=d)
        df = df.style.set_table_styles([
            {'selector': '.row_heading, .blank',
             'props': [('display', 'none;')]
            }
        ])
        return df    
    
        
class EvaluationOfFit():
    def __init__(self, autoencoder, means_data,
                 means_pred_data=None,
                 plot_train=True, plot_test=True,
                 plot_pred=False, pdf_name="fit_plots.pdf",
                 q="Not provided", out_fc="Not provided", data_origin=" "):
        self.q = q
        self.fc = out_fc
        self.data_origin = data_origin
        self.autoenc = autoencoder
        self.means_data = means_data
        self.means_pred_data = means_pred_data
        self.plot_train = plot_train
        self.plot_test = plot_test
        self.plot_pred = plot_pred
        self.pdf_name = pdf_name
        self.set_plot()
        
    def set_plot(self):
        with PdfPages(self.pdf_name) as self.pdf:
            self.rgba = self.get_color()
            self.get_loss_values()
            if self.plot_train:
                self.title = "train"
                self.pred = self.autoenc.predicted_train
                self.means = self.means_data.splited_data.train
                self.call_plots()
            if self.plot_test:
                self.title = "test"
                self.pred = self.autoenc.predicted_test
                self.means = self.means_data.splited_data.test
                self.call_plots()
            if self.plot_pred:
                self.title = "unseen"
                self.pred = self.autoenc.predicted
                self.means = self.means_pred_data.splited_data.train
                self.call_plots()
                
    def get_loss_values(self):    
        fig = plt.figure()
        fig.suptitle('Last average loss values after training', fontsize=16,fontweight='bold')
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.75)
        #
        ax.set_title('Middle layer q = %s\nOutlier fold change fc = %s\n%s'%(self.q,self.fc,self.data_origin), fontsize=14)

        ax.text(1, 3, 'loss: %0.2f' % self.autoenc.history.history["loss"][-1], style='italic',
                bbox={'facecolor':'lightgrey', 'alpha':0.5, 'pad':10}, fontsize=15)

        ax.text(8, 3, 'val_loss: %0.2f' % self.autoenc.history.history["val_loss"][-1], style='italic',
                bbox={'facecolor':'lightgrey', 'alpha':0.5, 'pad':10}, fontsize=15)

        ax.axis([0, 16, 0, 5])
        plt.axis('off')
        plt.show()
        self.pdf.savefig(fig)
        plt.close()
        
    def call_plots(self):
        self.plot_true_vs_pred_mean_10samples()
        self.get_coefs()
        self.hist_rho()
        self.plot_rho_sample("worst")
        self.plot_rho_sample("best")
        self.hist_slope_val_of_regression()
        self.plot_slope_sample("worst")
        self.plot_slope_sample("best")
            
    def get_color(self, idx=0):
        cmap = cm.get_cmap('Set3')
        rgba = cmap(idx)
        return rgba
    
    def plot_true_vs_pred_mean_10samples(self):
        fig, ax = plt.subplots(figsize=(16,8))
        ax.set_position([0.1,0.1,0.5,0.8])
        colors = cm.Set3(np.linspace(0, 1, 10))
        colors =iter(colors)
        for i in range(0,10):
            ax.scatter(self.pred[i],
                        self.means[i],
                        label="Sample No.{}".format(i+1),
                        color=next(colors))
            ax.plot(self.means[i],
                     self.means[i],'k-') 
            ax.set_xlabel("Predicted means")
            ax.set_ylabel("True means")
            plt.title("Prediction of means by autoencoder. ("+self.title+")")
            ax.legend(loc=(1.04,0))
        plt.yscale('log')
        plt.xscale('log')    
        plt.show()
        self.pdf.savefig(fig)
        plt.close()
            
    def get_coefs(self):
        self.slope = np.empty([self.pred.shape[0]-1])
        self.intercept = np.empty([self.pred.shape[0]-1])
        self.r_value = np.empty([self.pred.shape[0]-1])
        self.p_value = np.empty([self.pred.shape[0]-1])
        self.std_err = np.empty([self.pred.shape[0]-1])
        for i in range(0, self.pred.shape[0]-1):
            self.slope[i], self.intercept[i],\
            self.r_value[i], self.p_value[i],\
            self.std_err[i] = sp.stats.linregress(self.pred[i], self.means[i])
        
    def hist_rho(self):
        fig, ax = plt.subplots(figsize=(16,8))
        ax.set_position([0.1,0.1,0.5,0.8])
        self.r_value = self.r_value[~np.isnan(self.r_value)]
        ax.hist(self.r_value, color=self.rgba)
        plt.title("Distribution of Spearman corr. coef.\nbetween true and predicted means, in "+self.title+" dataset.")
        ax.set_xlabel("Spearman correlation coeficient value")
        ax.set_ylabel("Frequency")
        ax.text(1.04,0.04,'Median: %0.2f' % np.median(self.r_value),
                    transform=ax.transAxes, fontsize=15)
        plt.show()
        self.pdf.savefig(fig)
        plt.close()
        
    def hist_slope_val_of_regression(self):
        fig, ax = plt.subplots(figsize=(16,8))
        ax.set_position([0.1,0.1,0.5,0.8])
        self.slope = self.slope[~np.isnan(self.slope)]
        ax.hist(self.slope, color=self.rgba)
        plt.title("Distribution of slope of regression\nbetween true and predicted means, in "+self.title+" dataset.")
        ax.set_xlabel("Value of regression line slope")
        ax.set_ylabel("Frequency")
        ax.text(1.04,0.04,'Median: %0.2f' % np.median(self.slope),
                    transform=ax.transAxes, fontsize=15)
        plt.show()
        self.pdf.savefig(fig)
        plt.close()
       
    def get_worst_rho(self):
        return np.min(self.r_value)
    
    def get_best_rho(self):
        return np.max(self.r_value)
    
    def get_worst_slope_value_idx(self):
        return np.argmax(np.abs(np.subtract(self.slope,1)))
    
    def get_best_slope_value_idx(self):
        return np.argmin(np.abs(np.subtract(self.slope,1)))
    
    def get_worst_slope_value(self):
        idx = self.get_worst_slope_value_idx()
        return self.slope[idx]

    def plot_rho_sample(self, title2):
        fig, ax = plt.subplots(figsize=(16,8))
        ax.set_position([0.1,0.1,0.5,0.8])
        if title2 == "best":
            ind = np.argmax(self.r_value)
            rho = self.get_best_rho()
        elif title2 == "worst":
            ind = np.argmin(self.r_value)
            rho = self.get_worst_rho()
        for i in range(ind,ind+1):
            ax.text(1.04,0.12,'R value: %0.2f' % rho,
                    transform=ax.transAxes, fontsize=15)
            ax.scatter(self.pred[i],
                        self.means[i], label="Sample No.{}".format(i), c=self.rgba)
            ax.plot(self.means[i], self.means[i],'k-', label='Ideal slope')
            ax.set_xlabel("Predicted means")
            ax.set_ylabel("True means")
            plt.title("Prediction of means by autoencoder ("+self.title+").\n Sample with "+title2+" Spearman cor. coef. value. ")
            ax.legend(loc=(1.04,0))
        plt.show()
        self.pdf.savefig(fig)
        plt.close()
        
    def plot_slope_sample(self, title2):
        fig, ax = plt.subplots(figsize=(16,8))
        ax.set_position([0.1,0.1,0.5,0.8])
        if title2 == "best":
            ind = self.get_best_slope_value_idx()
        elif title2 == "worst":
            ind = self.get_worst_slope_value_idx()
        for i in range(ind,ind+1):
            ax.text(1.04,0.18,'Slope of\nregression\nline: %0.2f' % self.slope[i],
                    transform=ax.transAxes, fontsize=15)
            #t=np.repeat(1,self.pred[i].shape[0])
            ax.scatter(self.pred[i], self.means[i],
                       label="Sample No.{}".format(i),
                       c=self.rgba, cmap=plt.cm.Set3)
            ax.plot(self.means[i], self.means[i],'k-', label='Ideal slope')
            ax.plot(self.pred[i], self.intercept[i] + self.slope[i]*self.pred[i],
                    'chartreuse', label='Regression line')
            ax.set_xlabel("Predicted means")
            ax.set_ylabel("True means")
            plt.title("Prediction of means by autoencoder ("+self.title+").\n Sample with "+title2+" slope of regression line. ")
            ax.legend(loc=(1.04,0))
        plt.show()
        self.pdf.savefig(fig)
        plt.close()      
        
class Evaluation():
    def __init__(self, counts=None, outlier_idx=None, pred_mu=None,
                 pred_dispersion=None, prediction_table=None, estimate_each=True,
                 estimate_all_genes_together=True, is_theta=False,
                 use_adj_pval=True, compute_one_sided=False,
                 q="Not provided", out_fc="Not provided", data_origin=" ",
                 pdf_name="p_eval.pdf"):
        self.q = q
        self.fc = out_fc
        self.data_origin = data_origin
        self.estimate_all = estimate_all_genes_together
        self.counts_ini = counts
        self.outlier_idx_ini = outlier_idx
        self.mu_ini = pred_mu
        self.dispersion_ini = pred_dispersion
        self.param_prediction_table = prediction_table
        self.param_is_theta = is_theta
        self.param_use_adj_pval = use_adj_pval
        self.compute_one_sided = compute_one_sided
        self.out_gene_idx = self.get_genes_with_out_idx()
        self.out_sample_idx = self.get_samples_with_out_idx()
        self.set_init_vectors()
        self.recompute_outlier_table()
        self.tp = self.compute_true_positives()
        self.fp = self.compute_false_positives()
        self.rp = self.get_real_positives()
        self.get_fn_tn_tp_fp()
        with PdfPages(pdf_name) as self.pdf:
            self.metric_table()
            if estimate_all_genes_together:
                if self.tp_nr !=0 and self.fp_nr !=0: 
                    self.tpr = self.compute_tpr()
                    self.fpr = self.compute_fpr()
                    self.plot_roc()
                self.plot_nr_of_true_outliers()
            self.plot_predicted_outliers(to=self.fp_nr+100,title="All genes, "+"subset with "+str(self.fp_nr+100)+" samples out of "+str(self.outlier_table.shape[0]))
            self.plot_true_outliers(to=self.fp_nr+100,title="All genes, "+"subset  with "+str(self.fp_nr+100)+" samples out of "+str(self.outlier_table.shape[0]))
            self.plot_against_expected_pvals(title="All genes, all samples")
            if estimate_each:
                self.estimate_all = False
                print("Genes with outliers: \n")
                print(self.out_gene_idx[0])
                for i in range(self.out_gene_idx[0].shape[0]):
                    self.set_init_vectors(gene_idx=self.out_gene_idx[0][i])
                    self.recompute_outlier_table()
                    self.plot_predicted_outliers(title="Gene with idx ",
                                                 idx=self.out_gene_idx[0][i])
                    self.plot_true_outliers(title="Gene with idx ",
                                            idx=self.out_gene_idx[0][i])
                    self.plot_against_expected_pvals(title="Gene with idx ",
                                            idx=self.out_gene_idx[0][i])
                #print("Samples with outliers: \n")
                #print(self.out_sample_idx[0])
                #for i in range(self.out_sample_idx[0].shape[0]):
                    #self.set_init_vectors_sample(sample_idx=self.out_sample_idx[0][i])
                    #self.recompute_outlier_table()
                    #self.plot_predicted_outliers(title="Sample with idx ",idx=self.out_sample_idx[0][i])
                    #self.plot_true_outliers(title="Sample with idx ",idx=self.out_gene_idx[0][i])            
            
    def get_genes_with_out_idx(self):
        s =np.sum(self.outlier_idx_ini, axis=0)
        genes_with_out = np.where(s)
        return(genes_with_out)
    
    def get_samples_with_out_idx(self):
        s =np.sum(self.outlier_idx_ini, axis=1)
        samples_with_out = np.where(s)
        return(samples_with_out)

    def set_init_vectors(self, gene_idx=None):
        if gene_idx is None:
            gene_idx = self.out_gene_idx[0][0]
        if self.estimate_all:
            self.counts = self.counts_ini.flatten()
            self.dispersion = self.dispersion_ini.flatten()
            self.mu = self.mu_ini.flatten()
            self.outlier_idx = self.outlier_idx_ini.flatten()
        else:
            print(gene_idx)
            self.counts = self.counts_ini[:,gene_idx]
            self.dispersion = self.dispersion_ini[:,gene_idx]
            self.mu = self.mu_ini[:,gene_idx]
            self.outlier_idx = self.outlier_idx_ini[:,gene_idx]
            
    def set_init_vectors_sample(self, sample_idx=None):
        if sample_idx is None:
            sample_idx = self.out_sample_idx[0][0]
        self.counts = self.counts_ini[sample_idx]
        self.dispersion = self.dispersion_ini[sample_idx]
        self.mu = self.mu_ini[sample_idx]
        self.outlier_idx = self.outlier_idx_ini[sample_idx]

    def compute_two_sided_p_val_theta(self):
        '''If theta provided'''
        p_vals = []
        #shape = mu.shape
        for x_ij,theta_ij,mu_ij in zip(self.counts,
                                       self.dispersion, self.mu):
            cdf_val = sp.stats.nbinom.cdf(k=x_ij, n=1/theta_ij, p=(1/theta_ij)/mu_ij+(1/theta_ij) )
            pmf_at_x_ij = sp.stats.nbinom.pmf(k=x_ij, n=1/theta_ij, p=(1/theta_ij)/mu_ij+(1/theta_ij) )
            p_val = min(cdf_val, 1-cdf_val+pmf_at_x_ij, 0.5)*2
            p_vals.append(p_val)
        p_vals = np.asarray(p_vals)  #.reshape(shape)
        return p_vals
    
    def compute_two_sided_p_val(self):
        '''If dispersion provided'''
        p_vals = []
        #shape = mu.shape
        for x_ij,disp_ij,mu_ij in zip(self.counts,
                                       self.dispersion, self.mu):
            cdf_val = sp.stats.nbinom.cdf(k=x_ij, n=disp_ij, p=disp_ij/(mu_ij+disp_ij) )
            #print("------\n")
            #print("k: ",x_ij)
            #print("mu: ",mu_ij)
            #print("cdf: ",cdf_val)
            
            pmf_at_x_ij = sp.stats.nbinom.pmf(k=x_ij, n=disp_ij, p=disp_ij/(mu_ij+disp_ij) )
            p_val = min(cdf_val, 1-cdf_val+pmf_at_x_ij, 0.5)*2
            #print("pval ", p_val)
            p_vals.append(p_val)
        p_vals = np.asarray(p_vals)  #.reshape(shape)
        return p_vals
    
    def compute_one_sided_p_val(self):
        '''If dispersion provided'''
        p_vals = []
        #shape = mu.shape
        for x_ij,disp_ij,mu_ij in zip(self.counts,
                                       self.dispersion, self.mu):
            p_val = sp.stats.nbinom.cdf(k=x_ij, n=disp_ij, p=disp_ij/(mu_ij+disp_ij) )
            p_vals.append(p_val)
        p_vals = np.asarray(p_vals)  #.reshape(shape)
        return p_vals
    
    def p_adjust_bh(self):
        p_vals = self.p_vals
        by_descend = p_vals.argsort()[::-1]
        by_orig = by_descend.argsort()
        steps = float(len(p_vals)) / np.arange(len(p_vals), 0, -1)
        q = np.minimum(1, np.minimum.accumulate(steps * p_vals[by_descend]))
        return q[by_orig]
    
    def p_adjust_r_bh(self):
        #stats = importr('stats')
        #p_adjust = stats.p_adjust(robjects.FloatVector(self.p_vals), method = 'BH')
        #np.array(p_adjust)
        return np.array(self.p_vals)
    
    def evaluate_prediction(self, pvals):#########!!!!!!!!!!!!!!!!!!!! replace p_vals_adj
        check_if_true_out = lambda x: 1 if x < 0.05 else 0
        prediction_table = np.vstack( (pvals, list(map(check_if_true_out, pvals)) )).T
        return prediction_table
        
    def prepare_outlier_table(self):
        #self.outlier_idx = self.outlier_idx.flatten()
        table_x = np.concatenate((self.prediction_table,
                                self.outlier_idx.reshape(self.outlier_idx.shape[0],1)), axis=1)
        table = np.concatenate((table_x,
                                self.counts.reshape(self.outlier_idx.shape[0],1)), axis=1)
        self.outlier_table = table[table[:,0].argsort()]
        return self.outlier_table
    
    def prepare_outlier_table_adj(self):
        #self.outlier_idx = self.outlier_idx.flatten()
        table_x = np.concatenate((self.prediction_table_adj,
                                self.outlier_idx.reshape(self.outlier_idx.shape[0],1)), axis=1)
        table = np.concatenate((table_x,
                                self.counts.reshape(self.outlier_idx.shape[0],1)), axis=1)
        self.outlier_table = table[table[:,0].argsort()]
        return self.outlier_table
    
    def recompute_outlier_table(self):
        if self.param_prediction_table is None:
            if self.param_is_theta == True:
                print("Parameter is theta")
                #self.p_vals = self.compute_two_sided_p_val_theta()
            else:
                if self.compute_one_sided:
                    self.p_vals = self.compute_one_sided_p_val()
                else:
                    self.p_vals = self.compute_two_sided_p_val()
            self.p_vals_adj = self.p_adjust_bh()
            #self.p_vals_adj_r = self.p_adjust_r_bh()
            if self.param_use_adj_pval:
                self.prediction_table = self.evaluate_prediction(self.p_vals_adj)
            else:
                self.prediction_table = self.evaluate_prediction(self.p_vals)
        else:
            self.prediction_table = self.param_prediction_table            
        self.outlier_table = self.prepare_outlier_table()
    
    def get_real_positives(self):
        return self.outlier_table[:,2] 
    
    def compute_true_positives(self):
        comp_tp = lambda x: 1 if x[1] == x[2] == 1 else 0
        tp = list(map(comp_tp, self.outlier_table)) 
        return tp
    
    def compute_false_positives(self):
        comp_fp = lambda x: 1 if x[1] == 1 and x[2] == 0 else 0
        fp = list(map(comp_fp, self.outlier_table))
        return fp
    
    def get_fn_tn_tp_fp(self):
        self.tn_nr, self.fp_nr, self.fn_nr, self.tp_nr = confusion_matrix(self.outlier_table[:,2], self.outlier_table[:,1]).ravel()

    def compute_fpr(self):
        fpr = np.cumsum(self.fp) / max(np.cumsum(self.fp)) #np.repeat(np.count_nonzero(self.outlier_table[:,1]==0), len(self.fp)) 
        return fpr
    
    def compute_tpr(self):
        tpr = np.cumsum(self.tp) / max(np.cumsum(self.tp)) #np.repeat(np.count_nonzero(self.outlier_table[:,1]), len(self.tp))
        return tpr
    
    def plot_roc(self):
        roc_auc = auc(self.fpr, self.tpr)
        fig = plt.figure()
        fig.suptitle('Receiver operating characteristic')
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.8)
        ax.set_title('Middle autoencoder layer q = %s\nOutliers injected with fold change fc = %s'%(self.q,self.fc), fontsize=14)
        ax.plot(self.fpr, self.tpr, color='darkorange',
                 lw=2, label='ROC curve(area = %0.2f)' % roc_auc)
        #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        #plt.xlim([0.0, 1.0])
        #plt.ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()
        self.pdf.savefig(fig)
        plt.close()
        #plt.gcf().clear()
    
    def plot_nr_of_true_outliers(self):
        cumulative_true_outliers = np.cumsum(self.rp)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Middle autoencoder layer q = %s\nOutliers injected with fold change fc = %s'%(self.q,self.fc), fontsize=14)
        ax.plot(range(0,cumulative_true_outliers.shape[0]), cumulative_true_outliers)
        ax.set_ylabel('Number of true outliers')
        ax.set_xlabel('Samples sorted according to p-values')
        plt.title('Middle autoencoder layer q = %s\nOutliers injected with fold change fc = %s'%(self.q,self.fc))
        plt.show()
        self.pdf.savefig(fig)
        plt.close()
        #plt.gcf().clear()
        
    def plot_predicted_outliers(self,to=None,title=" ",idx=" "):
        if to is None:
            to = self.outlier_table.shape[0]
        fig = plt.figure()
        fig.suptitle('%s'%(str(title)+str(idx)),fontsize=16)
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.8)
        ax.set_title('Middle autoencoder layer q = %s\nOutliers injected with fold change fc = %s'%(self.q,self.fc), fontsize=14)
        ax.scatter(range(0,len(self.outlier_table[0:to,3])),
                        self.outlier_table[0:to,3],
                        c=self.outlier_table[0:to,1], alpha=0.5,
                       label="Predicted as outlier")
        ax.set_xlabel("sroted according pval")
        ax.set_ylabel("Counts")
        plt.yscale("log")
        plt.legend(loc="upper right", markerscale=0.7, scatterpoints=1, fontsize=10)
        ax = plt.gca()
        leg = ax.get_legend()
        leg.legendHandles[0].set_color('yellow')
        plt.show()
        self.pdf.savefig(fig)
        plt.close()
        #plt.gcf().clear()
        
    def plot_true_outliers(self,to=None,title=" ",idx=" "):
        if to is None:
            to = self.outlier_table.shape[0]
        fig = plt.figure()
        fig.suptitle('%s'%(str(title)+str(idx)),fontsize=16)
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.8)
        ax.set_title('Middle autoencoder layer q = %s\nOutliers injected with fold change fc = %s'%(self.q,self.fc), fontsize=14)
        ax.scatter(range(0,len(self.outlier_table[0:to,3])),
                        self.outlier_table[0:to,3],
                        c=self.outlier_table[0:to,2], alpha=0.5,
                       label="True outliers")
        ax.set_xlabel("sroted according pval")
        ax.set_ylabel("Counts")
        plt.yscale("log")
        plt.legend(loc="upper right", markerscale=0.7, scatterpoints=1, fontsize=10)
        ax = plt.gca()
        leg = ax.get_legend()
        leg.legendHandles[0].set_color('yellow')
        plt.show()
        self.pdf.savefig(fig)
        plt.close()
        
    def plot_against_expected_pvals(self,title=" ",idx=" "):
        fig = plt.figure()
        fig.suptitle('%s'%(str(title)+str(idx)),fontsize=16)
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.8)
        ax.set_title('Middle autoencoder layer q = %s\nOutliers injected with fold change fc = %s'%(self.q,self.fc), fontsize=14)
        expected = np.arange(1,self.outlier_table[:,0].shape[0]+1)/self.outlier_table[:,0].shape[0]
        ax.scatter(self.outlier_table[:,0], expected)
        ax.set_xlabel("p-values")
        ax.set_ylabel("expected p-values")
        plt.show()
        self.pdf.savefig(fig)
        plt.close()
        
    def metric_table(self):
        self.recall = self.tp_nr / (self.tp_nr + self.fp_nr)
        self.precision = self.tp_nr / (self.tp_nr + self.fn_nr)
        self.accuracy = (self.tp_nr + self.tn_nr) / (self.tp_nr + self.tn_nr + self.fn_nr + self.fp_nr) 
        
        fig = plt.figure()
        fig.suptitle('Metrics', fontsize=16,fontweight='bold')
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.75)
        ax.set_title('Middle autoencoder layer q = %s\nOutliers injected with fold change fc = %s\n%s'%(self.q,self.fc,self.data_origin), fontsize=14)
        ax.text(1, 7, 'Accuracy: %0.2f' % self.accuracy, style='italic',
                bbox={'facecolor':'lightgrey', 'alpha':0.5, 'pad':10}, fontsize=15)

        ax.text(1, 5, 'Precision: %0.2f' % self.precision, style='italic',
                bbox={'facecolor':'lightgrey', 'alpha':0.5, 'pad':10}, fontsize=15)

        ax.text(1, 3, 'Recall: %0.2f' % self.recall, style='italic',
                bbox={'facecolor':'lightgrey', 'alpha':0.5, 'pad':10}, fontsize=15)

        ax.text(9, 7, 'Confussion table: ', style='italic', fontsize=15)

        ax.text(15, 5, 'TP: %.0f' % int(self.tp_nr), style='italic',
                bbox={'facecolor':'papayawhip', 'alpha':0.5, 'pad':10}, fontsize=15)

        ax.text(9, 3, 'FN: %.0f' % int(self.fn_nr), style='italic',
                bbox={'facecolor':'papayawhip', 'alpha':0.5, 'pad':10}, fontsize=15)

        ax.text(15, 3, 'TN: %.0f' % int(self.tn_nr), style='italic',
                bbox={'facecolor':'papayawhip', 'alpha':0.5, 'pad':10}, fontsize=15)

        ax.text(9, 5, 'FP: %.0f' % int(self.fp_nr), style='italic',
                bbox={'facecolor':'papayawhip', 'alpha':0.5, 'pad':10}, fontsize=15)
        ax.axis([0, 19, 0, 9])
        plt.axis('off')
        plt.show()
        self.pdf.savefig(fig)
        plt.close()


class EvaluationTable():
    def __init__(self, counts=None, outlier_idx=None, pred_mu=None,
                 pred_dispersion=None, prediction_table=None, estimate_each=True,
                 estimate_all_genes_together=True, is_theta=False,
                 use_adj_pval=True, compute_one_sided=False,
                 q="Not provided", out_fc="Not provided", data_origin=" ",
                 pdf_name="p_eval.pdf"):
        self.q = q
        self.fc = out_fc
        self.data_origin = data_origin
        self.estimate_all = estimate_all_genes_together
        self.counts_ini = counts
        self.outlier_idx_ini = outlier_idx
        self.mu_ini = pred_mu
        self.dispersion_ini = pred_dispersion
        self.param_prediction_table = prediction_table
        self.param_is_theta = is_theta
        self.param_use_adj_pval = use_adj_pval
        self.compute_one_sided = compute_one_sided
        #self.rgba = self.get_color()
        self.out_gene_idx = self.get_genes_with_out_idx()
        self.out_sample_idx = self.get_samples_with_out_idx()
        self.set_init_vectors()
        self.recompute_outlier_table()
        with PdfPages(pdf_name) as self.pdf:
            if estimate_each:
                self.estimate_all = False
                print("Genes with outliers: \n")
                print(self.out_gene_idx[0])
                #for i in range(self.out_gene_idx[0].shape[0]):
                #    self.set_init_vectors(gene_idx=self.out_gene_idx[0][i])
                #    self.recompute_outlier_table()
                #    self.plot_predicted_outliers(title="Gene ",
                #                                 idx=self.out_gene_idx[0][i])
                #    self.plot_true_outliers(title="Gene ",
                #                            idx=self.out_gene_idx[0][i])
                #    self.plot_against_expected_pvals(title="Gene ",
                #                            idx=self.out_gene_idx[0][i])
                
                #print("Samples with outliers: \n")
                #print(self.out_sample_idx[0])
                #for i in range(self.out_sample_idx[0].shape[0]):
                    #self.set_init_vectors_sample(sample_idx=self.out_sample_idx[0][i])
                    #self.recompute_outlier_table()
                    #self.plot_predicted_outliers(title="Sample with idx ",idx=self.out_sample_idx[0][i])
                    #self.plot_true_outliers(title="Sample with idx ",idx=self.out_gene_idx[0][i])            
            
    
    def get_genes_with_out_idx(self):
        s =np.sum(self.outlier_idx_ini, axis=0)
        genes_with_out = np.where(s)
        return(genes_with_out)
    
    def get_samples_with_out_idx(self):
        s =np.sum(self.outlier_idx_ini, axis=1)
        samples_with_out = np.where(s)
        return(samples_with_out)

    def set_init_vectors(self, gene_idx=None):
        if gene_idx is None:
            gene_idx = self.out_gene_idx[0][0]
        if self.estimate_all:
            self.counts = self.counts_ini.flatten()
            self.dispersion = self.dispersion_ini.flatten()
            self.mu = self.mu_ini.flatten()
            self.outlier_idx = self.outlier_idx_ini.flatten()
        else:
            print(gene_idx)
            self.counts = self.counts_ini[:,gene_idx]
            self.dispersion = self.dispersion_ini[:,gene_idx]
            self.mu = self.mu_ini[:,gene_idx]
            self.outlier_idx = self.outlier_idx_ini[:,gene_idx]
            
    def set_init_vectors_sample(self, sample_idx=None):
        if sample_idx is None:
            sample_idx = self.out_sample_idx[0][0]
        self.counts = self.counts_ini[sample_idx]
        self.dispersion = self.dispersion_ini[sample_idx]
        self.mu = self.mu_ini[sample_idx]
        self.outlier_idx = self.outlier_idx_ini[sample_idx]

    def compute_two_sided_p_val_theta(self):
        '''If theta provided'''
        p_vals = []
        #shape = mu.shape
        for x_ij,theta_ij,mu_ij in zip(self.counts,
                                       self.dispersion, self.mu):
            cdf_val = sp.stats.nbinom.cdf(k=x_ij, n=1/theta_ij, p=(1/theta_ij)/mu_ij+(1/theta_ij) )
            pmf_at_x_ij = sp.stats.nbinom.pmf(k=x_ij, n=1/theta_ij, p=(1/theta_ij)/mu_ij+(1/theta_ij) )
            p_val = min(cdf_val, 1-cdf_val+pmf_at_x_ij, 0.5)*2
            p_vals.append(p_val)
        p_vals = np.asarray(p_vals)  #.reshape(shape)
        return p_vals
    
    def compute_two_sided_p_val(self):
        '''If dispersion provided'''
        p_vals = []
        #shape = mu.shape
        for x_ij,disp_ij,mu_ij in zip(self.counts,
                                       self.dispersion, self.mu):
            cdf_val = sp.stats.nbinom.cdf(k=x_ij, n=disp_ij, p=disp_ij/(mu_ij+disp_ij) )
            #print("------\n")
            #print("k: ",x_ij)
            #print("mu: ",mu_ij)
            #print("cdf: ",cdf_val)
            
            pmf_at_x_ij = sp.stats.nbinom.pmf(k=x_ij, n=disp_ij, p=disp_ij/(mu_ij+disp_ij) )
            p_val = min(cdf_val, 1-cdf_val+pmf_at_x_ij, 0.5)*2
            #print("pval ", p_val)
            p_vals.append(p_val)
        p_vals = np.asarray(p_vals)  #.reshape(shape)
        return p_vals
    
    def compute_one_sided_p_val(self):
        '''If dispersion provided'''
        p_vals = []
        #shape = mu.shape
        for x_ij,disp_ij,mu_ij in zip(self.counts,
                                       self.dispersion, self.mu):
            p_val = sp.stats.nbinom.cdf(k=x_ij, n=disp_ij, p=disp_ij/(mu_ij+disp_ij) )
            p_vals.append(p_val)
        p_vals = np.asarray(p_vals)  #.reshape(shape)
        return p_vals
    
    def p_adjust_bh(self):
        p_vals = self.p_vals
        by_descend = p_vals.argsort()[::-1]
        by_orig = by_descend.argsort()
        steps = float(len(p_vals)) / np.arange(len(p_vals), 0, -1)
        q = np.minimum(1, np.minimum.accumulate(steps * p_vals[by_descend]))
        return q[by_orig]
    
    def p_adjust_r_bh(self):
        #stats = importr('stats')
        #p_adjust = stats.p_adjust(robjects.FloatVector(self.p_vals), method = 'BH')
        #np.array(p_adjust)
        return np.array(self.p_vals)
    
    def evaluate_prediction(self, pvals):#########!!!!!!!!!!!!!!!!!!!! replace p_vals_adj
        check_if_true_out = lambda x: 1 if x < 0.05 else 0
        prediction_table = np.vstack( (pvals, list(map(check_if_true_out, pvals)) )).T
        return prediction_table
        
    def prepare_outlier_table(self):
        #self.outlier_idx = self.outlier_idx.flatten()
        table_x = np.concatenate((self.prediction_table,
                                self.outlier_idx.reshape(self.outlier_idx.shape[0],1)), axis=1)
        table = np.concatenate((table_x,
                                self.counts.reshape(self.outlier_idx.shape[0],1)), axis=1)
        self.outlier_table = table[table[:,0].argsort()]
        self.outlier_table_pd = pd.DataFrame({'Pvalue':self.outlier_table[:,0],
                                              'Predicted_id':self.outlier_table[:,1],
                                              'True_id':self.outlier_table[:,2],
                                              'log(Counts+1)':np.log1p(self.outlier_table[:,3])
                                             })
        self.outlier_table_pd['Predicted as outlier'] = np.where(self.outlier_table_pd['Predicted_id']==1, 'yes', 'no')
        self.outlier_table_pd['True outlier'] = np.where(self.outlier_table_pd['True_id']==1, 'yes', 'no')
        return self.outlier_table
    
    def prepare_outlier_table_adj(self):
        #self.outlier_idx = self.outlier_idx.flatten()
        table_x = np.concatenate((self.prediction_table_adj,
                                self.outlier_idx.reshape(self.outlier_idx.shape[0],1)), axis=1)
        table = np.concatenate((table_x,
                                self.counts.reshape(self.outlier_idx.shape[0],1)), axis=1)
        self.outlier_table = table[table[:,0].argsort()]
        self.outlier_table_pd = pd.DataFrame({'Pvalue':self.outlier_table[:,0],
                                              'Predicted':self.outlier_table[:,1],
                                              'True':self.outlier_table[:,2],
                                              'log(Counts+1)':np.log1p(self.outlier_table[:,3])
                                             })
        self.outlier_table_pd['Predicted as outlier'] = np.where(self.outlier_table_pd['Predicted_id']==1, 'yes', 'no')
        self.outlier_table_pd['True outlier'] = np.where(self.outlier_table_pd['True_id']==1, 'yes', 'no')
        return self.outlier_table
    
    def recompute_outlier_table(self):
        if self.param_prediction_table is None:
            if self.param_is_theta == True:
                print("Parameter is theta")
                #self.p_vals = self.compute_two_sided_p_val_theta()
            else:
                if self.compute_one_sided:
                    self.p_vals = self.compute_one_sided_p_val()
                else:
                    self.p_vals = self.compute_two_sided_p_val()
            self.p_vals_adj = self.p_adjust_bh()
            #self.p_vals_adj_r = self.p_adjust_r_bh()
            if self.param_use_adj_pval:
                self.prediction_table = self.evaluate_prediction(self.p_vals_adj)
            else:
                self.prediction_table = self.evaluate_prediction(self.p_vals)
        else:
            self.prediction_table = self.param_prediction_table            
        self.outlier_table = self.prepare_outlier_table()
    

class EvalPlot():
    def __init__(self, counts=None, outlier_idx=None, pred_mu=None,
                 pred_dispersion=None, prediction_table=None, estimate_one_gene=True,
                 estimate_all_genes_together=True, is_theta=False,
                 use_adj_pval=True, compute_one_sided=False,
                 q="Not provided", out_fc="Not provided", data_origin=" ",
                 pdf_name="p_eval.pdf", outlier_table=None, outlier_table_pd=None, gene_idx=None):
        self.q = q
        self.fc = out_fc
        self.data_origin = data_origin
        self.estimate_all = estimate_all_genes_together
        self.counts_ini = counts
        self.outlier_idx_ini = outlier_idx
        self.mu_ini = pred_mu
        self.dispersion_ini = pred_dispersion
        self.param_prediction_table = prediction_table
        self.param_is_theta = is_theta
        self.param_use_adj_pval = use_adj_pval
        self.compute_one_sided = compute_one_sided
        self.outlier_table = outlier_table
        self.outlier_table_pd = outlier_table_pd
        self.out_gene_idx = gene_idx
        self.tp = self.compute_true_positives()
        self.fp = self.compute_false_positives()
        self.rp = self.get_real_positives()
        self.get_fn_tn_tp_fp()
        with PdfPages(pdf_name) as self.pdf:
            self.metric_table()
            if estimate_all_genes_together:
                if self.tp_nr !=0 and self.fp_nr !=0: 
                    self.tpr = self.compute_tpr()
                    self.fpr = self.compute_fpr()
                    self.plot_roc()
            if estimate_one_gene:
                self.nr=100
                self.plot_nr_of_true_outliers()
                self.plot_predicted_outliers(title="Gene ",
                                             idx=self.out_gene_idx)
                self.plot_true_outliers(title="Gene ",
                                        idx=self.out_gene_idx)
                self.plot_against_expected_pvals(title="Gene ",
                                        idx=self.out_gene_idx)
            else:
                self.nr=1000
                self.plot_nr_of_true_outliers()
                self.plot_predicted_outliers(to=self.fp_nr+100,title="All genes, "+"subset of "+str(self.fp_nr+100)+" samples out of "+str(self.outlier_table.shape[0]))
                self.plot_true_outliers(to=self.fp_nr+100,title="All genes, "+"subset of "+str(self.fp_nr+100)+" samples out of "+str(self.outlier_table.shape[0]))
                self.plot_against_expected_pvals(title="All genes, all samples")

        
    def get_color(self, idx=0):
        cmap = cm.get_cmap('Set3')
        rgba = cmap(idx)
        return rgba
    
    def get_real_positives(self):
        return self.outlier_table[:,2] 
    
    def compute_true_positives(self):
        comp_tp = lambda x: 1 if x[1] == x[2] == 1 else 0
        tp = list(map(comp_tp, self.outlier_table)) 
        return tp
    
    def compute_false_positives(self):
        comp_fp = lambda x: 1 if x[1] == 1 and x[2] == 0 else 0
        fp = list(map(comp_fp, self.outlier_table))
        return fp
    
    def get_fn_tn_tp_fp(self):
        self.tn_nr, self.fp_nr, self.fn_nr, self.tp_nr = confusion_matrix(self.outlier_table[:,2], self.outlier_table[:,1]).ravel()

    def compute_fpr(self):
        fpr = np.cumsum(self.fp) / max(np.cumsum(self.fp)) #np.repeat(np.count_nonzero(self.outlier_table[:,1]==0), len(self.fp)) 
        return fpr
    
    def compute_tpr(self):
        tpr = np.cumsum(self.tp) / max(np.cumsum(self.tp)) #np.repeat(np.count_nonzero(self.outlier_table[:,1]), len(self.tp))
        return tpr
    
    def plot_roc(self):
        roc_auc = auc(self.fpr, self.tpr)
        fig = plt.figure()
        fig.suptitle('Receiver operating characteristic')
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.8)
        ax.set_title('Middle autoencoder layer q = %s\nOutliers injected with fold change fc = %s'%(self.q,self.fc), fontsize=14)
        ax.plot(self.fpr, self.tpr, color='darkorange',
                 lw=2, label='ROC curve(area = %0.2f)' % roc_auc)
        #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        #plt.xlim([0.0, 1.0])
        #plt.ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()
        self.pdf.savefig(fig)
        plt.close()
        #plt.gcf().clear()
    
    def plot_nr_of_true_outliers(self):
        cumulative_true_outliers = np.cumsum(self.rp)[0:self.nr-1]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(str(self.nr)+' samples with smallest p-values, q = %s'%(self.q), fontsize=14) #\nOutliers injected with fold change fc = %s ,self.fc
        ax.plot(range(0,self.nr-1), cumulative_true_outliers)
        ax.set_ylabel('Number of true outliers')
        ax.set_xlabel('Sorted according to p-value')
        #plt.title('Middle autoencoder layer q = %s\nOutliers injected with fold change fc = %s'%(self.q,self.fc))
        plt.show()
        self.pdf.savefig(fig)
        plt.close()
        #plt.gcf().clear()
        
    def plot_predicted_outliers(self,to=None,title=" ",idx=" "):
        if to is None:
            to = self.outlier_table.shape[0]
        fig = plt.figure()
        fig.suptitle('%s'%(str(title)+str(idx)),fontsize=16)
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.85)
        ax.set_title('q = %s'%(self.q), fontsize=14) #\nOutliers injected with fold change fc = %s ,self.fc
        #ax.scatter(range(0,len(self.outlier_table[0:to,3])),
         #               self.outlier_table[0:to,3],
         #               c=self.outlier_table[0:to,1], alpha=0.5,
         #              label="Predicted as outlier")   
        ax.scatter(self.outlier_table[0:to,0],
                        np.log1p(self.outlier_table[0:to,3]),
                        c=self.outlier_table[0:to,1], alpha=0.5,
                       label="Predicted as outliers")
        ax.set_xlabel("sroted according pval")
        ax.set_ylabel("Counts")
        #plt.yscale("log")
        plt.legend(loc="upper right", markerscale=0.7, scatterpoints=1, fontsize=10)
        ax = plt.gca()
        leg = ax.get_legend()
        leg.legendHandles[0].set_color('yellow')
        plt.show()
        self.pdf.savefig(fig)
        plt.close()
        #plt.gcf().clear()
        p = ggplot(aes(x='Pvalue', y='log(Counts+1)', color='Predicted as outlier'), data=self.outlier_table_pd)  + geom_point(size=100) +\
            labs(title=str(title)+str(idx), x="p-values") + geom_vline(x=[0.01, 0.05], color="black")# + theme(legend.text=element_text(size=20))#+ scale_y_log() #+\
            #theme(legend.text=element_text(size=20))
        p
        print(p)
        
    def plot_true_outliers(self,to=None,title=" ",idx=" "):
        if to is None:
            to = self.outlier_table.shape[0]
        fig = plt.figure()
        fig.suptitle('%s'%(str(title)+str(idx)),fontsize=16)
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.85)
        ax.set_title('q = %s'%(self.q), fontsize=14) #\nOutliers injected with fold change fc = %s ,self.fc
        #ax.scatter(range(0,len(self.outlier_table[0:to,3])),
         #               self.outlier_table[0:to,3],
          #              c=self.outlier_table[0:to,2], alpha=0.5,
           #            label="True outliers")
        ax.scatter(self.outlier_table[0:to,0],
                        np.log1p(self.outlier_table[0:to,3]),
                        c=self.outlier_table[0:to,2], alpha=0.5,
                       label="True outliers")
        ax.set_xlabel("sroted according pval")
        ax.set_ylabel("Counts")
        #plt.yscale("log")
        plt.legend(loc="upper right", markerscale=0.7, scatterpoints=1, fontsize=10)
        ax = plt.gca()
        leg = ax.get_legend()
        leg.legendHandles[0].set_color('yellow')
        plt.show()
        self.pdf.savefig(fig)
        plt.close()
        p = ggplot(aes(x='Pvalue', y='log(Counts+1)', color='True outlier'), data=self.outlier_table_pd)  + geom_point(size=100) +\
            labs(title=str(title)+str(idx), x="p-values") + geom_vline(x=[0.05], color="black")+ geom_vline(x=[0.01], color="black",linetype='dashed') # + theme(legend.text=element_text(size=20))#+ #geom_segment(aes(x=0.5, y=0, xend=0.5, yend=1))#geom_vline(xintercept = 1, size=3)#+\
            #theme(legend.text=element_text(size=20))
        p
        print(p)
        
    def plot_against_expected_pvals(self,title=" ",idx=" "):
        fig = plt.figure()
        fig.suptitle('%s'%(str(title)+str(idx)),fontsize=16)
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.8)
        ax.set_title('Middle autoencoder layer q = %s\nOutliers injected with fold change fc = %s'%(self.q,self.fc), fontsize=14)
        expected = np.arange(1,self.outlier_table[:,0].shape[0]+1)/self.outlier_table[:,0].shape[0]
        ax.scatter(self.outlier_table[:,0], expected)
        ax.set_xlabel("p-values")
        ax.set_ylabel("expected p-values")
        plt.show()
        self.pdf.savefig(fig)
        plt.close()
        
    def metric_table(self):
        self.recall = self.tp_nr / (self.tp_nr + self.fp_nr)
        self.precision = self.tp_nr / (self.tp_nr + self.fn_nr)
        self.accuracy = (self.tp_nr + self.tn_nr) / (self.tp_nr + self.tn_nr + self.fn_nr + self.fp_nr) 
        
        fig = plt.figure()
        fig.suptitle('Metrics', fontsize=16,fontweight='bold')
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.75)
        ax.set_title('Middle autoencoder layer q = %s\nOutliers injected with fold change fc = %s\n%s'%(self.q,self.fc,self.data_origin), fontsize=14)
        ax.text(1, 7, 'Accuracy: %0.2f' % self.accuracy, style='italic',
                bbox={'facecolor':'lightgrey', 'alpha':0.5, 'pad':10}, fontsize=15)

        ax.text(1, 5, 'Precision: %0.2f' % self.precision, style='italic',
                bbox={'facecolor':'lightgrey', 'alpha':0.5, 'pad':10}, fontsize=15)

        ax.text(1, 3, 'Recall: %0.2f' % self.recall, style='italic',
                bbox={'facecolor':'lightgrey', 'alpha':0.5, 'pad':10}, fontsize=15)

        ax.text(9, 7, 'Confussion table: ', style='italic', fontsize=15)

        ax.text(15, 5, 'TP: %.0f' % int(self.tp_nr), style='italic',
                bbox={'facecolor':'papayawhip', 'alpha':0.5, 'pad':10}, fontsize=15)

        ax.text(9, 3, 'FN: %.0f' % int(self.fn_nr), style='italic',
                bbox={'facecolor':'papayawhip', 'alpha':0.5, 'pad':10}, fontsize=15)

        ax.text(15, 3, 'TN: %.0f' % int(self.tn_nr), style='italic',
                bbox={'facecolor':'papayawhip', 'alpha':0.5, 'pad':10}, fontsize=15)

        ax.text(9, 5, 'FP: %.0f' % int(self.fp_nr), style='italic',
                bbox={'facecolor':'papayawhip', 'alpha':0.5, 'pad':10}, fontsize=15)
        ax.axis([0, 19, 0, 9])
        plt.axis('off')
        plt.show()
        self.pdf.savefig(fig)
        plt.close()


















