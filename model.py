from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute
from keras.optimizers import RMSprop
import keras.backend as K

class DQNModel:
    def __init__(self, nb_actions=4, input_shape = (4, 84, 84)):
        model = Sequential()
        model.add(Permute((2, 3, 1), input_shape=input_shape))
        model.add(Conv2D(32, (8, 8), strides=(4, 4)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))
        model.compile(optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), loss='mse')
        self.model = model
        
    def summary(self):
        print(self.model.summary())
