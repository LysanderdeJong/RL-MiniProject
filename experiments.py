# local imports
from qlearning import q_learning, EpsilonGreedyPolicy

# Global imports
import numpy as np

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
    try:
        Q = np.zeros((environment.nS, environment.nA))
    except Exception:
        Q = np.zeros((environment.env.nS, environment.env.nA))
    policy = EpsilonGreedyPolicy(Q, epsilon=epsilon, double=double)
    Q_values, (episode_lengths, episode_returns) = q_learning(environment,
                                                              num_episodes,
                                                              policy,
                                                              discount_factor=discount_factor,
                                                              alpha=alpha)

    return Q_values, (episode_lengths, episode_returns)

    #todo: write logs to disk or otherwise perhaps plot them immediately already