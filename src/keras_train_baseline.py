from environments.billiards import *

# from sklearn.linear_model import
from keras.layers import Input, Dense, Reshape, Flatten, Permute, TimeDistributed, Activation, Lambda, multiply, subtract, concatenate
from keras.layers import SimpleRNN, LSTM, GRU, Conv2D, MaxPooling2D
from keras.models import Model
from keras.losses import kullback_leibler_divergence
from keras.regularizers import L1L2, Regularizer
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

latent_dim = 32      # Latent space dimensionality
reg_lambda = 1e-4    # Global regularization coefficient
decoder_loss = 1.0   # Weight of the decoder loss

env = Billiards()
X = []
v = []
M = 15
for i in range(M):
    env.make_steps()
    env.make_frames(32)
    X.append(env.frames)
    v.append(env.X)

X = np.stack(X)
X = X / np.max(X)
X = np.expand_dims(X, axis = 2)
v = np.stack(v)
v = v.reshape((v.shape[0], v.shape[1], np.prod(v.shape[2:])))
print('Shape of the image tensor: ' + str(X.shape))
print('Shape of the velocity tensor: ' + str(v.shape))

## Create a baseline correlator network
iD = Input(shape=X.shape[1:])
xD = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation = 'relu', data_format='channels_first'), input_shape=X.shape)(iD)
xD = TimeDistributed(MaxPooling2D())(xD)
xD = TimeDistributed(Conv2D(32, (3, 3), padding='same', activation = 'relu'))(xD)
xD = TimeDistributed(MaxPooling2D())(xD)
xD = TimeDistributed(Flatten())(xD)
xD = LSTM(32, return_sequences = True)(xD)
xD = Dense(v.shape[2], activation = 'linear')(xD)
BaselineCorrelator = Model(inputs=iD, outputs=[xD], name='BaselineCorrelator')
BaselineCorrelator.compile(optimizer='rmsprop', loss=['mse'])
BaselineCorrelator.summary()
history = BaselineCorrelator.fit(X, v, epochs=1024, batch_size = 15)
print(BaselineCorrelator.evaluate(X, v))
