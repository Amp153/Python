# Implement Q learning using an RBF network
from __future__ import print_function, division
from builtins import range

import gym
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor


class FeatureTransformer:
    def __init__(self, env):
        # Gather 10,000 samples from state space
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        # standardize observations so they have a mean 0 and varience 1
        scaler = StandardScaler()
        scaler.fit(observation_examples)
        
        # Used to convert a state to a features representation
        # We use RBF kernals with different variances to cover different parts of
        # number of components is the number of exemplars
        featurizer = FeatureUnion([
                ("rbf1", RBFSampler(gamma=5.0, n_components=500)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=500)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=500)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=500))
                ])
        featurizer.fit(scaler.transform(observation_examples))
        # Making them instance variables so I can put them in tranform function
        self.scaler = scaler
        self.featurizer = featurizer
        
    def transform(self, observations):
        # print "observations:", observations
        scaled = self.scaler.transform(observations)
        # assert(len(scaled.shape) == 2)
        return self.featurizer.transform(scaled)

# holds one SGDRegressor for each action
class Model:
    def __init__(self,env,feature_transformer,learning_rate):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate=learning_rate)
            # partial fit target 0
            model.partial_fit(feature_transformer.transform([env.reset]),[0])
            self.models.append(model)
        
    # turns state into feature vector and makes a prediction of values
    # returned as a numpy array, s is in a list before we call transform
    # its because theta has to be 1d in scikit learn
    def predict(self,s):
        X = self.feature_transformer.transform([s])
        assert(len(X.shape) == 2)
        return np.array([m.predict(X)[0] for m in self.models])
    
    # scikit learn wants targets to be 1d too
    def update(self,s,a,G):
        X = self.feature_transformer.transform([s])
        assert(len(X.shape) == 2)
        self.models[a].partial_fit(X,[G])
        
    # implements epsilon greedy, we dont need it because SGDRegressor predits 0
    # for all states until they are updated. This works as a optimistic initial
    # values method since all the rewards are -1
    def sample_action(self,s,eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))
    
# returns a list of states_and_rewards, and the total reward
# model instance, epsilon greedy, gamma = discount rate
def play_one(model,eps,gamma):
    # reset the environment to initials some var
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done and iters < 10000:
        action = model.sample_action(observation,eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)
            
        # update the model using the q learning equation
        G = reward + gamma*np.max(model.predict(observation)[0])
        model.update(prev_observation,action,G)
        
        totalreward += reward
        iters += 1

    return totalreward

# This is a plot of the negative of the optimal value function, it's plotable because
# this state is 2d so we can make a 3d plot
def plot_cost_to_go(env,estimator,num_tiles=20):
    x = np.linspace(env.observation_space.low[0],env.observation_space.high[0]),
    y = np.linspace(env.observation_space.low[1],env.observation_space.high[1]),
    X,Y = np.meshgrid(x,y)
    # both X and Y will be of shape (num_tiles,num_tiles)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack)
    #X is also of shape (num_tiles,num_tiles)
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111,projection='3d')
    surf = ax.plot_surface(X,Y,Z,rstride=1,cstride=1,
                           cmap=matplotlib.cm.coolwarm,vmin=-1.0,vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-To-Go == -V(s)')
    ax.set_title("Cost-To-Go Function")
    fig.colorbar(surf)
    plt.show()

# Running average is how it's scored in open ai gym
# returns for each episode will vary alot so running avg, ai is judged for how
# well it does over 100 episodes
def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0,t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft, "constant")
    # learning rate = 10e-5
    # eps = 1.0
    gamma = 0.99
    
    # monitor is optional and command line argument
    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)
        
    N = 300
    totalrewards = np.empty(N)
    for n in range(N):
        eps = 0.1*(0.97**n)
        totalreward = play_one(model,eps,gamma)
        totalrewards[n] = totalreward
        print("episode:",n,"total reward:",totalreward)
    print("avg reward for last 100 episodes:",totalrewards[-100:].mean())
    print("total steps:",totalrewards.sum())
    
    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()
    
    plot_running_avg(totalrewards)

    # plot the optimal state-value function
    plot_cost_to_go(env,model)