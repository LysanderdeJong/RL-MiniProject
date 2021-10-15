import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class BiasMDP(gym.Env):
    """Environment for example MDP from Sutton & Barto chapter 6.7."""
    
    def __init__(self, seed = None):
        super(BiasMDP, self).__init__()
        self.observation_space = spaces.Discrete(3)
        self.action_space = spaces.Discrete(10) # Should be even number!
        self.seed()
        self.reset()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        assert(self.action_space.contains(action))
        
        reward = 0
        if self.state == 0:
            if action % 2 == 1:
                self.state = 2
            else:
                self.state = 1
        elif self.state == 1:
            reward = self.np_random.normal(-0.1, 1)
            self.state = 2
        return self.state, reward, self.state == 2, {}
    
    def reset(self):
        self.state = 0
        return self.state
