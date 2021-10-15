# local imports
from qlearning import q_learning, EpsilonGreedyPolicy

# Global imports
import numpy as np
import gym

def run_experiment(environment,
                   num_episodes,
                   discount_factor=1.0,
                   alpha=0.5,
                   epsilon=0.1,
                   double=False,
                   seed=None):

    # load desired environment
    if seed:
        environment.seed(seed)
    
    Q_dim = []
    if hasattr(environment, 'nS'):
        Q_dim = [environment.nS, environment.nA]
    else:
        if isinstance(environment.observation_space, gym.spaces.Tuple):
            for d in environment.observation_space.spaces:
                Q_dim.append(d.n)
            Q_dim.append(environment.action_space.n)
        else:
            Q_dim = [environment.observation_space.n, environment.action_space.n]
    Q = np.zeros(Q_dim)
    
    policy = EpsilonGreedyPolicy(Q, epsilon=epsilon, double=double)
    Q_values, (episode_lengths, episode_returns) = q_learning(environment,
                                                              num_episodes,
                                                              policy,
                                                              discount_factor=discount_factor,
                                                              alpha=alpha)

    return Q_values, (episode_lengths, episode_returns)

    #todo: write logs to disk or otherwise perhaps plot them immediately already
