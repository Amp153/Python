import gym
import numpy as np

env = gym.make('CartPole-v0')
#Resets the environment, returns 4 floating point numbers, cart position, velocity, pole angle, velocity at tip
env.reset()

#box = env.observation_space
#env.action_space #ranges from 0 to 1

#info is only used for debugging and shouldn't be used (but it'll definately improve the ai)
#observation, reward, done, info = env.step(action)
done = False
while not done:
    observation, reward, done, _ = env.step(env.action_space.sample())
    
#Will end quickly because random actions
    
#Goal right now: determine the average of how many steps are taken when actions are randomly sampled
#This is a benchmark
    
    
#state.dot(params) > 0 -> do action 1; if < 0 -> do action 0
    
'''
For # of times I want to adjust the weights
    new_weights = random
    For # episodes I want to play to decide whether to update the weights
        play episode
    if avg episode length > best so far:
        weights = new_weights
play a final set of episodes to see how good my best weights do again
'''
best = 0
new_weights = np.random.random(4)
avg_length = episodes_played / 2

if avg_length > best:
    weights = new_weights