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

def linear_objective(y_true, y_pred):
    return 1.0 - 1.0 * K.mean(y_pred, axis=-1)

def linear_regularizer(x):
    return K.mean(x)

def mse_regularizer(x):
    return K.mean(K.square(x))

## Define Extra Layers to Use
class ArbitraryRegularization(Layer):
    def __init__(self, regularizer_function, **kwargs):
        super(ArbitraryRegularization, self).__init__(**kwargs)
        self.supports_masking = True
        self.activity_regularizer = regularizer_function
    def get_config(self):
        config = {'regularizer_function': regularizer_function}
        base_config = super(ArbitraryRegularization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ApplyRegularization(Layer):
    def __init__(self, **kwargs):
        super(ApplyRegularization, self).__init__(**kwargs)
    def build(self, input_shape):
        super(ApplyRegularization, self).build(input_shape)
    def call(self, x):
        return x[0]

class CELayer(Layer):
    def __init__(self, **kwargs):
        super(CELayer, self).__init__(**kwargs)
    def build(self, input_shape):
        super(CELayer, self).build(input_shape)
    def call(self, x):
        return -1. * kullback_leibler_divergence(x[0], x[1])
    def compute_output_shape(self, input_shape):
        return (None, X.shape[1]-1)

class CPLayer(Layer):
    def __init__(self, **kwargs):
        super(CPLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        super(CPLayer, self).build(input_shape)
    def call(self, x):
            y_true = K.l2_normalize(x[0], axis=-1)
            y_pred = K.l2_normalize(x[1], axis=-1)
            return K.sum(y_true * y_pred, axis=-1)
    def compute_output_shape(self, input_shape):
        return (None, X.shape[1]-1)

class SoftmaxDropout(Layer):
    def __init__(self, mean, stddev, **kwargs):
        super(SoftmaxDropout, self).__init__(**kwargs)
        self.supports_masking = True
        self.mean = mean
        self.stddev = stddev

    def call(self, x, training=None):
        return K.cast(K.greater(x + K.random_normal(shape=K.shape(x),
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
    x_gru = GRU(latent_dim, return_sequences = True)(i)
    x = TimeDistributed(Dense(X.shape[2], activation='sigmoid'))(x_gru)
    xa_probabilities = x
    x = SoftmaxDropout(0.,1.0)(x)
    xa = Reshape([X.shape[1], 32])(x)
    x = TimeDistributed(Dense(X.shape[2], activation='sigmoid'))(x_gru)
    xb_probabilities = x
    x = SoftmaxDropout(0.,1.0)(x)
    xb = Reshape([X.shape[1], 32])(x)
    model = Model(inputs=i, outputs=[xa, xb, xa_probabilities, xb_probabilities], name='Consciousness')
    return model

def generator_rnn():
    ia = Input(shape=(X.shape[1], latent_dim))
    ib = Input(shape=(X.shape[1], latent_dim))
    ic = Input(shape=(X.shape[1], latent_dim))
    i = concatenate([ia, ib, ic])
    x = GRU(latent_dim, return_sequences = True)(i)
    model = Model(inputs=[ia, ib, ic], outputs=x, name='Generator')
    return model

def decoder_rnn():
    i = Input(shape=(X.shape[1], latent_dim))
    x = GRU(1024, return_sequences = True)(i)
    x = TimeDistributed(Reshape((X.shape[2], X.shape[3])))(x)
    model = Model(inputs=i, outputs=x, name='Decoder')
    return model

print(X.shape)
## Start constructing the circuit
i = Input(shape=X.shape[1:])
R = representation_rnn()
C = consciousness_rnn()
G = generator_rnn()
D = decoder_rnn()

h = R(i) # Get h from R
c_A, c_B, c_A_soft, c_B_soft = C(h) # Get masks c_A and c_B from C
b = multiply([h, c_B], name = 'b') # Get b through elementwise multiplication
a_hat = G([c_A, c_B, b]) # Send c_A, c_B and b to G to get a_hat

a_hat = Lambda(lambda x: x[:,:-1,:], output_shape=(X.shape[1]-1, latent_dim))(a_hat) # Slice dimensions to align vectors
h_A = Lambda(lambda x: x[:,1:,:], output_shape=(X.shape[1]-1, latent_dim))(h) # Slice dimensions to align vectors
c_A = Lambda(lambda x: x[:,:-1,:], output_shape=(X.shape[1]-1, latent_dim))(c_A) # Slice dimensions to align vectors

h_A = multiply([h_A, c_A]) # Calculate h[A] to compare against a_hat
a_hat = multiply([a_hat, c_A]) # Mask a_hat
consciousness_error = subtract([a_hat, h_A])
consciousness_error = ArbitraryRegularization(mse_regularizer, name='Consciousness_Generator_Error')(consciousness_error)

b_transformed = Dense(latent_dim, activation='linear')(b) # Create a layer that attempts to make b independent from h[A]
b_transformed = Lambda(lambda x: x[:,:-1,:], output_shape=(X.shape[1]-1, latent_dim))(b_transformed)
b_transformed = multiply([b_transformed, c_A])
transformation_error = subtract([b_transformed, h_A])
transformation_error = ArbitraryRegularization(mse_regularizer, name='Transformation_Error')(transformation_error)

intelligence_error = concatenate([c_A_soft, c_B_soft]) # The more elements we choose to predict, the more "intelligent" we are
intelligence_error = Flatten()(intelligence_error)
intelligence_error = ArbitraryRegularization(linear_regularizer, name='Intelligence_Level')(intelligence_error)

x_hat = D(a_hat)
x_hat = ApplyRegularization()([x_hat, consciousness_error, transformation_error, intelligence_error])

## Compile the model and start training
CN = Model(inputs=i, outputs=[x_hat])
CN.compile(optimizer='rmsprop', loss=['mse'])
CN.summary()
history = CN.fit(X, [X[:,1:,:]], epochs=1024, validation_split = 0.25)
