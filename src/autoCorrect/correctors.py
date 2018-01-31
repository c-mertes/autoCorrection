from abc import abstractmethod
from .autoencoder import Autoencoder
from .data_utils import TrainTestPreparation, ZeroInjectionWhereMean, FoldChInjectionWhereMean, DataCookerWithPred
from keras.optimizers import Adam, RMSprop
import numpy as np

class Corrector():        
    @abstractmethod
    def correct(self, counts, size_factors, **kwargs):
        pass


class DummyCorrector(Corrector):
    def __init__(self):
        pass

    def correct(self, counts, size_factors, **kwargs):
        return np.ones_like(self.counts)


class AECorrector(Corrector):
    def __init__(self, parameters=None, denoisingAE=False,
                 inject_zeros=True, epochs=2, encoding_dim=10, lr=0.001):
        self.denoisingAE = denoisingAE
        #self.parameters = parameters
        self.epochs = epochs
        self.encoding_dim=encoding_dim
        self.lr = lr
        self.inject_zeros = inject_zeros
        
    def correct(self, counts, size_factors, **kwargs):
        self.loader = DataCookerWithPred(counts, inject_outliers=self.denoisingAE,
                                          pred_counts=counts)# size_factors,
        self.data = self.loader.data()
        self.ae = Autoencoder(choose_autoencoder=True,
                              encoding_dim=self.encoding_dim,
                              size=counts.shape[1])
        self.ae.model.compile(optimizer=Adam(lr=self.lr), loss=self.ae.loss)
        self.ae.model.fit(self.data[0][0], self.data[0][1],  
                            epochs=self.epochs,
                            batch_size=None,
                            shuffle=True,
                            validation_data=(self.data[1][0], self.data[1][1]),
                            verbose=0
                           )
        self.corrected = self.ae.model.predict(self.data[2][0])
        return self.corrected



    

        
        