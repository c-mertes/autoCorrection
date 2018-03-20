from abc import abstractmethod
from .autoencoder import Autoencoder
from .data_utils import DataCooker
from .layers import ConstantDispersionLayer
from keras.optimizers import Adam, RMSprop
from keras.models import model_from_json
import numpy as np
import json
import os
dir, filename = os.path.split(__file__)
MODEL_PATH = os.path.join(dir,"..","..","saved_models")
OPT_PARAM_PATH = os.path.join(dir,"..","..","saved_models", "best")

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
    def __init__(self, model_name="model", model_directory=MODEL_PATH, verbose=1,
                 param_path=OPT_PARAM_PATH, param_exp_name=None, denoisingAE=True,
                 save_model=True, epochs=250, encoding_dim=23, lr=0.00068, batch_size=None):
        self.denoisingAE = denoisingAE
        self.save_model = save_model
        self.model_name = model_name
        self.directory = model_directory
        self.verbose = verbose
        if param_exp_name is not None:
            path = os.path.join(param_path,param_exp_name+"_best.json")
            metrics = json.load(open(path))
            self.batch_size = metrics['batch_size']
            self.epochs = metrics['epochs']
            self.encoding_dim = metrics['encoding_dim']
            self.lr = metrics['lr']
        else:
            self.epochs = epochs
            self.encoding_dim = encoding_dim
            self.lr = lr
            self.batch_size = batch_size

    def correct(self, counts, size_factors=None, only_predict=False):
        if len(counts.shape) == 1:
            counts = counts.reshape(1,counts.shape[0])
            size_factors = size_factors.reshape(1,size_factors.shape[0])
        if size_factors is not None and counts.shape != size_factors.shape:
            raise ValueError("Size factors and counts must have equal dimensions."+
                             "\nNow counts shape:"+str(counts.shape)+ \
                            "\nSize factors shape:"+str(size_factors.shape))
        if not os.path.isfile(os.path.join(self.directory,self.model_name+'.json')) and only_predict:
            raise ValueError("There is no model "+str(os.path.join(self.directory,self.model_name+'.json'))+
                  "' saved. Only predict is not possible!")
        self.loader = DataCooker(counts, size_factors,
                                 inject_outliers=self.denoisingAE,
                                 only_prediction=only_predict)
        self.data = self.loader.data()
        if not only_predict:
            self.ae = Autoencoder(choose_autoencoder=True,
                                  encoding_dim=self.encoding_dim,
                                  size=counts.shape[1])
            self.ae.model.compile(optimizer=Adam(lr=self.lr), loss=self.ae.loss)
            self.ae.model.fit(self.data[0][0], self.data[0][1],
                                epochs=self.epochs,
                                batch_size=self.batch_size,
                                shuffle=True,
                                validation_data=(self.data[1][0], self.data[1][1]),
                                verbose=self.verbose
                               )
            if self.save_model:
                model_json = self.ae.model.to_json()
                with open(os.path.join(self.directory,self.model_name+'.json'), "w") as json_file:
                    json_file.write(model_json)
                self.ae.model.save_weights(os.path.join(self.directory,self.model_name+'_weights.h5'))
                print("Model saved on disk!")
            model = self.ae.model
        else:
            json_file = open(os.path.join(self.directory,self.model_name+'.json'), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json,
                    custom_objects={'ConstantDispersionLayer': ConstantDispersionLayer})
            model.load_weights(os.path.join(self.directory,self.model_name+'_weights.h5'))
            print("Model loaded from disk!")
        self.corrected = model.predict(self.data[2][0])
        return self.corrected






