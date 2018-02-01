from abc import abstractmethod
from .autoencoder import Autoencoder
from .data_utils import TrainTestPreparation, ZeroInjectionWhereMean, FoldChInjectionWhereMean, DataCookerWithPred
from keras.optimizers import Adam, RMSprop
import numpy as np
import json

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
    def __init__(self, param_path=None, denoisingAE=False,
                 inject_zeros=True, epochs=2, encoding_dim=10, lr=0.001):
        if param_path is not None:
            metrics = json.load(open(param_path))
            self.denoisingAE = metrics['inj_out'] #metrics['param']['data']['inject_outliers']
            self.inject_zeros = metrics['inj_z'] #metrics['param']['data']['inject_zeros']
            self.epochs = metrics['epochs'] #metrics['param']['fit']['epochs']
            self.encoding_dim = metrics['m_emb'] #metrics['param']['model']['encoding_dim']
            self.lr = metrics['m_lr'] #metrics['param']['model']['lr']
        else
            self.denoisingAE = denoisingAE
            self.inject_zeros = inject_zeros
            self.epochs = epochs
            self.encoding_dim=encoding_dim
            self.lr = lr

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



    

        
        