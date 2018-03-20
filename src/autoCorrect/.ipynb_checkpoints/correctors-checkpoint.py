from abc import abstractmethod
from .autoencoder import Autoencoder
from .data_utils import DataCooker
from .layers import ConstantDispersionLayer
from keras.optimizers import Adam, RMSprop
from keras.models import model_from_json
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
    def __init__(self, model_name="model", model_directory="/s/project/scared/model",
                 param_path=None, denoisingAE=True,
                 save_model=True, epochs=10, encoding_dim=23, lr=0.00068, batch_size=None):
        if param_path is not None:
            metrics = json.load(open(param_path))
            self.denoisingAE = metrics['inj_out'] 
            self.epochs = metrics['epochs']
            self.encoding_dim = metrics['m_emb'] 
            self.lr = metrics['m_lr'] 
        else:
            self.denoisingAE = denoisingAE
            self.epochs = epochs
            self.encoding_dim = encoding_dim
            self.lr = lr
            self.batch_size = batch_size
            self.save_model = save_model
            self.model_name = model_name
            self.directory = model_directory

    def correct(self, counts, size_factors, only_predict=False):
        if len(counts.shape) == 1:
            counts = counts.reshape(1,counts.shape[0])
            size_factors = size_factors.reshape(1,size_factors.shape[0])
        if counts.shape != size_factors.shape:
            raise ValueError("Size factors and counts must have equal dimensions."+
                             "\nNow counts shape:"+str(counts.shape)+ \
                            "\nSize factors shape:"+str(size_factors.shape))
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
                                verbose=1
                               )
            if self.save_model:
                model_json = self.ae.model.to_json()
                with open(self.directory+'/'+self.model_name+'.json', "w") as json_file: 
                    json_file.write(model_json)
                self.ae.model.save_weights(self.directory+'/'+self.model_name+'_weights.h5')
                print("Model saved on disk!")
            model = self.ae.model
        else:
            json_file = open(self.directory+'/'+self.model_name+'.json', 'r') 
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json,
                                            custom_objects={'ConstantDispersionLayer': ConstantDispersionLayer})
            model.load_weights(self.directory+'/'+self.model_name+'_weights.h5')
            print("Model loaded from disk!")
        self.corrected = model.predict(self.data[2][0])
        return self.corrected



    

        
        