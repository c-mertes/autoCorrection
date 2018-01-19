from keras.layers import Input, Dense, Lambda, Multiply
from keras.models import Model
from keras.callbacks import EarlyStopping
from .loss import NB
from .layers import ConstantDispersionLayer
from keras import backend as K
import matplotlib.pyplot as mp
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt
from .myCallbacks import PlotLosses
from keras import losses
import tensorflow as tf
import numpy as np
import scipy as sp
import h5py



class Autoencoder():
    def __init__(self, train_in=None, sf_train=None, test_in=None,
                 sf_test=None, train_out=None, test_out=None,
                 predict_data=None, sf_predict=None, means_data=None, 
                 encoding_dim=2, size=10000, optimizer=Adam(),
                 choose_autoencoder=False,
                 choose_encoder=False, choose_decoder=False, epochs=1100):
        self.train_in = train_in
        self.sf_train = sf_train
        self.test_in = test_in
        self.sf_test = sf_test
        self.train_out = train_out
        self.test_out = test_out
        self.predict_data = predict_data
        self.sf_predict = sf_predict
        self.means_data = means_data
        self.val_losses = []
        self.encoding_dim = encoding_dim
        self.size = size
        self.epochs = epochs
        self.ClippedExp = lambda x: K.minimum(K.exp(x), 1e5)
        self.Invert = lambda x: K.pow(x, -1) 
        self.choose_autoencoder = choose_autoencoder
        self.choose_encoder = choose_encoder
        self.choose_decoder = choose_decoder
        self.autoenc_model = self.get_autoencoder()
        self.model = self.set_model()
        self.optimizer = optimizer
        self.loss = self.set_loss()
        self.compile_model()
        self.sess = tf.Session()
        K.set_session(self.sess)
        self.run_session()
        self.fit_model()
        #self.current_val_loss = self.get_current_val_loss()
        #self.append_val_loss()
        if not choose_encoder:
            self.predicted_dispersion = self.get_dispersion()
        self.predicted_test = self._predict_test()
        self.predicted_train = self._predict_train()
        if self.predict_data is not None:
            self.predicted = self._predict()


    def get_autoencoder(self, encoding_dim=None):
        if encoding_dim != None:
            self.encoding_dim = encoding_dim
        self.input_layer = Input(shape=(self.size,), name='inp')
        self.sf_layer = Input(shape=(self.size,), name='sf')
        self.normalized = Multiply()([self.input_layer, self.sf_layer])
        encoded = Dense(self.encoding_dim, name='encoder', use_bias=True)(self.normalized)
        decoded = Dense(self.size, name='decoder', use_bias=True)(encoded)
        mean_scaled = Lambda(self.ClippedExp, output_shape=(self.size,), name="mean_scaled")(decoded)
        inv_sf = Lambda(self.Invert, output_shape=(self.size,), name="inv")(self.sf_layer)
        mean = Multiply()([mean_scaled, inv_sf])
        self.disp = ConstantDispersionLayer(name='dispersion')
        self.output = self.disp(mean)
        self.model = Model([self.input_layer, self.sf_layer], self.output)
        return self.model
    
    def get_encoder(self):
        self.encoder = Model([self.autoenc_model.get_layer('inp').input, 
                              self.autoenc_model.get_layer('sf').input],
                             self.autoenc_model.get_layer('encoder').output)
        return self.encoder

    def get_decoder(self):
        encoded_input = Input(shape=(self.encoding_dim,))
        decoder_layer = self.autoenc_model.get_layer('decoder')
        decoded = decoder_layer(encoded_input)
        mean_layer = self.autoenc_model.get_layer('mean')
        mean = mean_layer(decoded)
        dispersion_layer = ConstantDispersionLayer(name='dispersion')
        output = dispersion_layer(mean)
        self.decoder = Model(encoded_input, output)
        return self.decoder

    def set_model(self):
        if self.choose_autoencoder:
            self.model = self.get_autoencoder()
        elif self.choose_encoder:
            self.model = self.get_encoder()
        elif self.choose_decoder:
            self.model = self.get_decoder()
        return self.model

    def set_loss(self):
        if self.choose_autoencoder:
            nb = NB(self.model.get_layer('dispersion').theta)
            self.loss = nb.loss
        elif self.choose_encoder:
            self.loss = losses.mean_squared_error
        elif self.choose_decoder:
            nb = NB(self.model.get_layer('dispersion').theta)
            self.loss = nb.loss
        return self.loss


    def compile_model(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def run_session(self):
        self.sess.run(tf.global_variables_initializer())

    def fit_model(self):
        self.plot_losses = PlotLosses()
        #stoping = EarlyStopping(monitor='val_loss',
        #                      min_delta=0,
        #                      patience=20,
        #                      verbose=0, mode='min')
        with self.sess.as_default():
            self.history = self.model.fit([self.train_in, self.sf_train], self.train_out,  # x_train_log1p, x_train,
                                          epochs=self.epochs,
                                          batch_size=None,
                                          shuffle=True,
                                          validation_data=([self.test_in,self.sf_test], self.test_out),
                                          callbacks=[self.plot_losses,],# stoping],
                                          verbose=0
                                          )

    def get_current_val_loss(self):
        with self.sess.as_default():
            self.val_loss = self.loss(self.test_in, self.test_out).eval()
        return self.val_loss
    
    def get_pred_val_loss(self):
        with self.sess.as_default():
            self.val_loss = self.loss(self.pred_data, self.pred_data).eval()
        return self.val_loss

    def _predict_test(self):
        self.predicted_test = self.model.predict([self.test_in,self.sf_test])
        return self.predicted_test
    
    def _predict_train(self):
        self.predicted_train = self.model.predict([self.train_in,self.sf_train])
        return self.predicted_train
    
    def _predict(self):
        self.predicted = self.model.predict([self.predict_data, self.sf_predict])
        return self.predicted

    def append_val_loss(self):
        self.val_loss = self.get_current_val_loss()
        self.val_losses.append((self.val_loss, self.encoding_dim))

    def get_val_losses(self):
        return self.val_losses

    def get_dispersion(self):
        with self.sess.as_default():
            self.predicted_dispersion = self.model.get_layer('dispersion').theta.eval()
        return self.predicted_dispersion
    
    def get_corr_for_test(self):
        rho_test = np.empty([self.predicted_test.shape[0]-1])
        pval_test = np.empty([self.predicted_test.shape[0]-1])
        for i in range(0,self.predicted_test.shape[0]-1):
            rho_test[i], pval_test[i]=sp.stats.spearmanr(self.means_data.splited_data.test[i],
                                                         self.predicted_test[i])
        rho_test = rho_test[~np.isnan(rho_test)]
        return rho_test
        
    def get_corr_for_train(self):    
        rho_train = np.empty([self.predicted_train.shape[0]-1])
        pval_train = np.empty([self.predicted_train.shape[0]-1])
        for i in range(0,self.predicted_train.shape[0]-1):
            rho_train[i], pval_train[i]=sp.stats.spearmanr(self.means_data.splited_data.train[i],
                                                           self.predicted_train[i])
        rho_train = rho_train[~np.isnan(rho_train)]
        return rho_train

    def fit_on_different_hidden_dim(self, list_of_hidden_dim=(1, 2, 3)):
        # ,5,10,30,100,300,500,1000,10000)
        self.reset()
        fig1, ax1 = mp.subplots()
        for i in list_of_hidden_dim:
            print(i)
            self.model = self.get_autoencoder(encoding_dim=i)
            nb = NB(self.model.get_layer('dispersion').theta)
            self.loss = nb.loss
            self.compile_model()
            self.fit_model()
            self.append_val_loss()
            
            self.predicted_train = self._predict_train()
            self.predicted_test = self._predict_test()
            
        self.val_losses_array = np.asarray(self.get_val_losses())
        print(self.val_losses_array)
        self.sorted_val_losses = self.val_losses_array[self.val_losses_array[:, 1].argsort()]
        ax1.plot(self.sorted_val_losses[:, 1], self.sorted_val_losses[:, 0])
        ax1.set_xlabel("hidden dimension q")
        ax1.set_ylabel("validation loss value")
        fig1.savefig("sorted_val_losses.png")
        self.reset()

    def reset(self):  # gal ir vidurkius
        self.val_losses = []
        self.run_session()
        