# AI 2018

# Importing the libraries
import os
import numpy as np
import gym
from gym import wrappers
import pybullet_envs

# Setting the Hyper Parameters

class Hp():
    
    def __init__(self):
        # Training loops, number of times we update the model
        self.nb_steps = 1000
        # Time the AI walks on the field
        self.episode_length = 1000
        # Shouldn't be too small or large
        self.learning_rate = 0.02
        # Weights
        self.nb_directions = 16
        # Keep the best directions
        self.nb_best_directions = 16
        # Best way to be certain of something in python
        assert self.nb_best_directions <= self.nb_directions
        # It's called Augmented 'Random' Search for a reason
        self.noise = 0.03
        # To test on multiple different environments
        self.seed = 1
        # The name
        self.env_name = 'HalfCheetahBulletEnv-v0'
        
# Normalizing the states
        
class Normalizer():
    
    def __init__(self, nb_inputs):
        # Counter, vector of nb_inputs values all initialized to zero
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        # Numerator of the variance
        self.mean_diff = np.zeros(nb_inputs)
        # Variance
        self.var = np.zeros(nb_inputs)
    
    # Self is so I can use the init values
    def observe(self, x):
        # It's still a vector
        self.n += 1.
        # We need to know the previous mean
        last_mean = self.mean.copy()
        # Online computation of the new mean
        self.mean += (x - self.mean) / self.n
        # Online computation of variance
        self.mean_diff += (x - last_mean) * (x - self.mean)
        # We don't want it to be zero so 1.0 = (1*10^-2)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)
        
    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std
    
# Building the AI
# The AI is a policy
class Policy():
    
    def __init__(self, input_size, output_size):
        # Matrix of weights
        self.theta = np.zeros((output_size, input_size))
        
    # Delta is the perturbations, by default it's none
    def evaluate(self, input, delta = None, direction = None):
        # This is the equation
        if direction is None:
            # Multiply the matrix of weights by the vector of inputs
            return self.theta.dot(input)
        # Else if
        elif direction == "positive":
            return (self.theta + hp.noise*delta).dot(input)
        else:
            return (self.theta - hp.noise*delta).dot(input)
        
    def sample_deltas(self):
        # Random normals, it's the same size as theta
        return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]
    
    # Method of finite differences,Gradient decent without the loss, optimize the reward
    # Rollouts = Reward of the positive, negative,the protobation
    def update(self, rollouts, sigma_r):
        # it's zero in the same format
        step = np.zeros(self.theta.shape)
        # r is reward, d is the protobation, this is the right of equation
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        # Left of the equation
        self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step

# Exploring the policy on one specific direction and over one episode
        
def explore(env, normalizer, policy, direction = None, delta = None):
    state = env.reset()
    done = False
    num_plays = 0.
    sum_rewards = 0
    # while done != true
    while not done and num_plays < hp.episode_length:
        # Normalizer()
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        # High rewards are 1, low are -1, it works best in that range
        reward = max(min(reward, 1), -1)
        sum_rewards += reward
        num_plays += 1
    return sum_rewards

# Training the AI
    
def train(env, policy, normalizer, hp):
    
    for step in range(hp.nb_steps):
        
        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions
        
        # Getting the positive rewards in the positive directions
        for k in range(hp.nb_directions):
            positive_rewards[k] = explore(env, normalizer, policy, direction = "positive", delta = deltas[k])
        
        # Getting the negative rewards in the negative/opposite directions
        for k in range(hp.nb_directions):
            negative_rewards[k] = explore(env, normalizer, policy, direction = "negative", delta = deltas[k])
        
        # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()
        
        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        # It's a dictionary
        scores = {k:max(r_pos, r_neg) for k, (r_pos,r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        # will sort it by the highest value
        # scores.keys is keys in dictionary, lambda means function
        # If index 3 is highest it'll be first
        order = sorted(scores.keys(), key = lambda x:scores[x])[0:hp.nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
        
        # Updating our policy
        policy.update(rollouts, sigma_r)
        
        # Printing the final reward of the policy after the update
        reward_evaluation = explore(env, normalizer, policy)
        print('Step: ', step, 'Reward: ', reward_evaluation)
        
# Running the main code
# On windows this makes a directory in the users instead of the programs
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')

hp = Hp()
np.random.seed(hp.seed)
env = gym.make(hp.env_name)
# Something about being able to watch the videos of the AI
env = wrappers.Monitor(env, monitor_dir, force = True)
nb_inputs = env.observation_space.shape[0]
nb_outputs = env.action_space.shape[0]
policy = Policy(nb_inputs, nb_outputs)
normalizer = Normalizer(nb_inputs)
train(env, policy, normalizer, hp)
