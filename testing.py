import gym
import numpy as np
from matplotlib import pyplot as plt
from numpy import uint8

from dataSet import dataSet as ds


def update_state(state, x):
    observation.state_to_greyscale(obs)
    x = observation.state_resize(greyscale_obs, 84, 84)
    return np.append(state[1:], np.expand_dims(x, 0), axis=0)



env = gym.make("Breakout-v4")
obs = env.reset()
observation = ds()

for step in range(1000): # 1000 steps max
    action = np.random.randint(0, 4)
    obs, reward, done, info = env.step(action)
    greyscale_obs = observation.state_to_greyscale(obs)
    re_image = observation.state_resize(greyscale_obs, 84, 84)
    state = np.empty((1,4),dtype=uint8)
    stacked_frames = update_state(state, re_image)
    print(stacked_frames)
    print(stacked_frames.shape)
    env.render()
    if done:
        img = env.render(mode='rgb_array')
        plt.imshow(img)
        plt.show()
        plt.imshow(greyscale_obs)
        plt.show()
        plt.imshow(re_image)
        plt.show()
        #plt.imshow(stacked_frames)
        #plt.show()
        # print(x)
        # print(y)


        print("The game is over in {} steps".format(step))
        break
env.close()

state = np.empty((4, 84, 84), dtype=uint8)

print(state[1:])