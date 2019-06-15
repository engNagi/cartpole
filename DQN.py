import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from pexpect import searcher_re

"""
breakout -actions_space = 4 val
['NOOP', 'FIRE', 'RIGHT', 'LEFT']
keys_to_action = ['0', '1', '2', '3']

breakout -observation_ = 1D-Array of size = 128
"""

class Deep_Q_Network(object):
    """
    This class impelements the Model of the Agent for our Deep Q network
    Session =
    """
    def __init__(self, session, num_actions, learning_rate = 1e-3, history_size = 4, screen_height = 84, screen_width = 84, gamma = 0.98):
        self.session = session
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.history_size = history_size
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.gamma = gamma

        self.build_prediction_network()
        self.build_target_network()
        self.build_training()

    def build_prediction_network(self):
        with tf.variable_scope("prediction_network"):
            self.states = tf.placeholder(tf.float32, shape=())



    def build_target_network(self):
        pass

    def build_training(self):
        pass



