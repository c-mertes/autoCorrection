import keras
from sklearn.metrics import roc_auc_score
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as mp
import numpy as np
import IPython.core.display as dsp

#sess = tf.InteractiveSession()

class MyCallback(keras.callbacks.Callback):
	def __init__(self): # define your keyword/params here
		self.y_pred = []
		self.val_data_mean = []

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		self.y_pred = self.model.predict(self.validation_data[0])
		print("on_batch_end: y_pred = ", self.y_pred[0])
		#self.val_data_mean = np.mean(self.validation_data[0], axis=0)
		#mp.scatter(self.y_pred[0], self.val_data_mean, marker="v")

		#print("on_epch_end: validation data = ", self.validation_data[0])

		#self.y_pred_exp = K.exp(self.y_pred)
		#print("on_epch_end: y_pr_exp = ", self.y_pred_exp[1])
		#self.y_pred_exp = tf.Print(self.y_pred_exp, [self.y_pred_exp], message="This is y_pred_exp: ")
		#self.y_pred_exp.eval()
        
        
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = mp.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        dsp.clear_output(wait=True)
        mp.plot(self.x, self.losses, label="loss")
        mp.plot(self.x, self.val_losses, label="validation loss")
        mp.yscale('log')
        mp.ylabel('Loss value')
        mp.xlabel('Epochs')
        #mp.legend()
        mp.savefig('AutoCorrectLoss.png');
        
    def on_train_end(self, logs={}):
        print("Last loss: ", self.losses[-1])
        print("Last validation loss: ", self.val_losses[-1])


