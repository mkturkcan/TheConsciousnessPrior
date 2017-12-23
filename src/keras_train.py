from environments.billiards import *

from keras.layers import Input, Dense, Reshape, Flatten, Permute, TimeDistributed, Activation, Lambda, multiply, subtract, concatenate
from keras.layers import SimpleRNN, LSTM, GRU
from keras.models import Model
from keras.losses import kullback_leibler_divergence
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

latent_dim = 32

env = Billiards()
X = []
M = 15
for i in range(M):
    env.make_steps()
    env.make_frames(32)
    X.append(env.frames)

X = np.stack(X)
X.shape

def linear_objective(y_true, y_pred):
    return -1 * K.mean(y_pred, axis=-1)

## Define Extra Layers to Use
class CELayer(Layer):
    def __init__(self, **kwargs):
        super(CELayer, self).__init__(**kwargs)
    def build(self, input_shape):
        super(CELayer, self).build(input_shape)
    def call(self, x):
        return kullback_leibler_divergence(x[0], x[1])
    def compute_output_shape(self, input_shape):
        return (None, X.shape[1]-1)

class SoftmaxDropout(Layer):
    def __init__(self, mean, stddev, **kwargs):
        super(SoftmaxDropout, self).__init__(**kwargs)
        self.supports_masking = True
        self.mean = mean
        self.stddev = stddev

    def call(self, x, training=None):
        return x * K.cast(K.greater(x + K.random_normal(shape=K.shape(x),
                                                        mean=self.mean,
                                                        stddev=self.stddev), 0.5), 'float32')

## Define the Representation, Consciousness and Generator models
def representation_rnn():
    i = Input(shape=X.shape[1:])
    x = TimeDistributed(Flatten())(i)
    x = GRU(latent_dim, return_sequences = True)(x)
    model = Model(inputs=i, outputs=x, name='Representation')
    return model

def consciousness_rnn():
    i = Input(shape=(X.shape[1], latent_dim))
    x = GRU(latent_dim, return_sequences = True)(i)
    x = TimeDistributed(Dense(X.shape[2], activation='sigmoid'))(x)
    x = SoftmaxDropout(0.,1.0)(x)
    xa = Reshape([X.shape[1], 32])(x)
    x = TimeDistributed(Dense(X.shape[2], activation='sigmoid'))(x)
    x = SoftmaxDropout(0.,1.0)(x)
    xb = Reshape([X.shape[1], 32])(x)
    model = Model(inputs=i, outputs=[xa, xb], name='Consciousness')
    return model

def generator_rnn():
    ia = Input(shape=(X.shape[1], latent_dim))
    ib = Input(shape=(X.shape[1], latent_dim))
    ic = Input(shape=(X.shape[1], latent_dim))
    i = concatenate([ia, ib, ic])
    x = GRU(latent_dim, return_sequences = True)(i)
    model = Model(inputs=[ia, ib, ic], outputs=x, name='Generator')
    return model

## Start constructing the circuit
i = Input(shape=X.shape[1:])
R = representation_rnn()
C = consciousness_rnn()
G = generator_rnn()

h = R(i) # Get h from R
c_A, c_B = C(h) # Get masks c_A and c_B from C
b = multiply([h, c_B], name = 'b') # Get b through elementwise multiplication
a_hat = G([c_A, c_B, b]) # Send c_A, c_B and b to G to get a_hat
intelligence_error = concatenate([c_A, c_B]) # The more elements we choose to predict, the more "intelligent" we are
a_hat = Lambda(lambda x: x[:,:-1,:], output_shape=(X.shape[1]-1, latent_dim))(a_hat) # Slice dimensions to align vectors
h_A = Lambda(lambda x: x[:,1:,:], output_shape=(X.shape[1]-1, latent_dim))(h) # Slice dimensions to align vectors
c_A = Lambda(lambda x: x[:,:-1,:], output_shape=(X.shape[1]-1, latent_dim))(c_A) # Slice dimensions to align vectors

h_A = multiply([h_A, c_A]) # Calculate h[A] to compare against a_hat
a_hat = multiply([a_hat, c_A]) # Mask a_hat
prediction_error = subtract([a_hat, h_A], name='Prediction_Error')

b_transformed = Dense(latent_dim, activation='linear')(b)
b_transformed = Lambda(lambda x: x[:,:-1,:], output_shape=(X.shape[1]-1, latent_dim))(b_transformed)
transformation_error = subtract([b_transformed, h_A], name='Transformation_Error')
intelligence_error = Flatten(name='Intelligence_Level')(intelligence_error)
#b = Lambda(lambda x: x[:,:-1,:], output_shape=(X.shape[1]-1, latent_dim))(b)
#c_B = Lambda(lambda x: x[:,:-1,:], output_shape=(X.shape[1]-1, latent_dim))(c_B)
#mi_error = CELayer(name='KL_Divergence')([a_hat, h])


## Compile the model and start training
CN = Model(inputs=i, outputs=[prediction_error, transformation_error, intelligence_error])
CN.compile(optimizer='rmsprop', loss=['mse', 'mse', linear_objective])
CN.summary()
history = CN.fit(X,
                 [np.zeros((X.shape[0], X.shape[1]-1, latent_dim)), np.zeros((X.shape[0], X.shape[1]-1, latent_dim)), np.zeros((X.shape[0], (X.shape[1]-1)*2*latent_dim))],
                 epochs=1024)
