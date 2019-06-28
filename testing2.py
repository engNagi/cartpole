import gym
import os
from DQN import Deep_Q_Network
import numpy as np

env = gym.make('CartPole-v0')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print(state_size)

batch_size = 32

n_episode = 100

OP_dir = 'cartpole_output/'

if not os.path.exists(OP_dir):
    os.mkdir(OP_dir)

agent = Deep_Q_Network(state_size, action_size)

done =False

for e in range(n_episode):
    state = env.reset()
    state = np.reshape(state, [1,state_size])

    for time in range(10000):
        env.render()
        # select an action based on epsilon given a the last state the selection action is played using take_action method
        #which select random action from a given action set
        action = agent.take_action(state)
        #observe reward and next state
        #done is a boolean flag to tell if it either the end of the game by lossing or the end of the episodo
        next_state, reward, done, info = env.step(action)
        #storing the reward equal to the returned reward from the environment if we are not done
        reward = reward if not done else -10
        #storing the received next state and saving it, the reshape is to transpose the next state to match our neural model
        next_state = np.reshape(next_state, [1, state_size])
        #store our experiences in experience_table (replay_memory)
        agent.store_experience(state, action,reward, next_state, done)
        #mapping the next state to be the current state
        state = next_state
        #If the game is finish before our episodes as the agent lost break
        if done:
            print ("episode:{}/{}, score:{:}".format(e, n_episode, time, agent.epsilon))
            break
        # sample a set of our experience replay according to the batch size
        if len(agent.experience_table) > batch_size :
                agent.replay(batch_size)

        if e % 50 == 0:
            agent.save(OP_dir + "weights" + "{:04d}.format(e)" + ".hdf5")
    env.close()