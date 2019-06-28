import random
from cgi import maxlen

import gym
import numpy as np
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os

"""
breakout -actions_space = 4 val
['NOOP', 'FIRE', 'RIGHT', 'LEFT']
keys_to_action = ['0', '1', '2', '3']

breakout -observation_ = 1D-Array of size = 128
"""

class Deep_Q_Network(object):

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        #experience Replay_table
        self.experience_table = deque(maxlen=5000)

        # discount future reward
        self.gamma = 0.9

        # exploration
        self.epsilon = 1.0
        # minimizing exploration
        self.minimize_epsilon = 0.995
        # minimum exploration
        self.minimum_epsilon = 0.01

        self.learning = 0.001

        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning))

        return model

    def store_experience(self, state, action, reward, next_state, done):
        self.experience_table.append((state, action, reward, next_state, done))

    def take_action(self, state):
        #takes a random number and checks if its less than epsilon takes a random explatory action
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        #else take action predicted by our model
        action_values = self.model.predict(state)
        # return predicted action that gives the maximum reward
        return np.argmax(action_values[0])

    def replay(self, batch_size):
        mini_batch = random.sample(self.experience_table, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                # pass the batch of the preprocessed state to policy network
                future_target = self.model.predict(state)
                future_target[0][action] = target

                self.model.fit(state, future_target, epochs=1, verbose=0)

            if self.epsilon > self.minimum_epsilon:
                self.epsilon *= self.minimize_epsilon

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)