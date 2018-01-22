from abc import abstractmethod
from .autoencoder import Autoencoder
from .data_utils import TrainTestPreparation, ZeroInjectionWhereMean, FoldChInjectionWhereMean
import numpy as np

class Corrector():        
    @abstractmethod
    def correct(self):
        pass


class DummyCorrector(Corrector):
    def __init__(self, counts=None, size_factors=None):
        self.counts = counts
        self.size_factors = size_factors
        self.corrected = self.correct()
    
    def get_size_factors(self):
        return self.size_factors

    def correct(self):
        return np.ones_like(self.counts)


class DAECorrector(Corrector):
    def __init__(self, counts=None, size_factors=None,
                 parameters=None, inject_zeros=True):
        #TODO
        #parameters as dictionary
        #reorganize autoencoder class
        self.counts = counts
        self.size_factors = size_factors
        self.parameters = parameters
        self.inject_zeros = inject_zeros
        self.injected_outliers = self.inject_outliers()
        self.prepare_data()
        self.corrected = self.correct()
        
    def inject_outliers(self):
        if self.inject_zeros:
            self.injected_outliers = ZeroInjectionWhereMean(
                self.counts, nr_of_out=800)
        else:
            self.injected_outliers = FoldChInjectionWhereMean(
                self.counts, fold=-5, nr_of_out=800)
        return self.injected_outliers
        
    def prepare_data(self):
        self.count_data = TrainTestPreparation(
            data=self.counts.astype(int), no_rescaling=False, ones_sf=True)
        self.out_data = TrainTestPreparation(
            data=self.injected_outliers.outlier_data.data_with_outliers.astype(int),
                                                 no_rescaling=False, ones_sf=True)
        #self.sf = TrainTestPreparation(self.size_factors, no_rescaling=True)
        
    def correct(self):
        self.ae = Autoencoder(self.out_data.splited_data.train,
                          self.out_data.splited_data.size_factor_train,
                          self.out_data.splited_data.test,
                          self.out_data.splited_data.size_factor_test,
                          self.count_data.splited_data.train,
                          self.count_data.splited_data.test,
                          predict_data=self.out_data.splited_data.test,## <----change hier to: self.counts
                          sf_predict=self.out_data.splited_data.size_factor_test, ## <---- here to: self.size_factors
                          choose_autoencoder=True, epochs=100, encoding_dim=10)

        return self.ae.predicted




    

        
        