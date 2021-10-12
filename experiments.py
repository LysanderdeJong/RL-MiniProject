# local imports
from qlearning import q_learning, update, EpsilonGreedyPolicy
from utils import load_environment

# Global imports
import numpy as np

def run_experiment(environment,
                   num_episodes,
                   discount_factor=1.0,
                   alpha=0.5,
                   epsilon=0.1,
                   double=False):

    # load desired environment
    env = load_environment(environment)

    policy_fn = EpsilonGreedyPolicy
    update_fn = update
    Q_values, (episode_lengths, episode_returns) = q_learning(env,
                                                              num_episodes,
                                                              policy_fn,
                                                              update_fn,
                                                              discount_factor=discount_factor,
                                                              alpha=alpha,
                                                              epsilon=epsilon,
                                                              double=double)

    return Q_values, (episode_lengths, episode_returns)

    #todo: write logs to disk or otherwise perhaps plot them immediately already