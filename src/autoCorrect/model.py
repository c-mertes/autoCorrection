from keras.optimizers import RMSprop, Adam
from autoCorrect import autoencoder

def model(train_data, lr=0.01,
          encoding_dim=128, batch_size=None):
    size = train_data[0]["inp"].shape[1]
    ae = autoencoder.Autoencoder(choose_autoencoder=True, size=size, 
                     encoding_dim=encoding_dim, batch_size=batch_size)
    ae.model.compile(optimizer=RMSprop(lr=lr), loss=ae.loss)#metrics=['eval.loss']
    model = ae.model
    return model
