from __future__ import print_function, division
from builtins import range

import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt

# s is state weight is w dot is dot product
def get_action(s,w):
    return 1 if s.dot(w) > 0 else 0

def play_one_episode(env,params):
    # render plays a video of it as it's being played (its slow)
    # env.render()
    # t is length of episode
    observation = env.reset()
    done = False
    t = 0
    
    while not done and t < 10000:
        t += 1
        action = get_action(observation,params)
        observation, reward, done, info = env.step(action)
        if done:
            break
        
    return t

# T is time to play, point is to return the average episode length
def play_multiple_episodes(env,T,params):
    episode_lengths = np.empty(T)
    
    for i in range(T):
        episode_lengths[i] = play_one_episode(env,params)
        
    avg_length = episode_lengths.mean()
    print("avg length:",avg_length)
    return avg_length

# Search through 100 random parameter vectors
# each are randomly selected between -1 and 1
def random_search(env):
    episode_lengths = []
    best = 0
    params = None
    for t in range(100):
        new_params = np.random.random(4)*2 - 1
        # T=100 so we test each parameter vector 100 times
        avg_length = play_multiple_episodes(env,100,new_params)
        episode_lengths.append(avg_length)
        
        if avg_length > best:
            params = new_params
            best = avg_length
    return episode_lengths, params

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, 'Desktop\Python\DeepReinforcementLearning\CartPole')
    episode_lengths, params = random_search(env)
    plt.plot(episode_lengths)
    plt.show()
    
    print("***Final run with final weights***")
    play_multiple_episodes(env, 100, params)