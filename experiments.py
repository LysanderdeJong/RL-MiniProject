# local imports
from qlearning import q_learning, EpsilonGreedyPolicy

# Global imports
import numpy as np
from collections import defaultdict
import random
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

def multi_trail_experiment(env_name, env, config):
    results = defaultdict(lambda: defaultdict(list))
    
    num_episodes = config["num_episodes"]
    num_trails = config["num_trails"]
    
    q_discount_factor = config["q_discount_factor"]
    q_alpha = config["q_alpha"]
    q_epsilon = config["q_epsilon"]
    
    dq_discount_factor = config["dq_discount_factor"]
    dq_alpha = config["dq_alpha"]
    dq_epsilon = config["dq_epsilon"]
    
    for i in range(num_trails):
        np.random.seed(i)
        random.seed(i)
        Q_values, (episode_lengths, episode_returns) = run_experiment( env,
                                                                       num_episodes,
                                                                       discount_factor=q_discount_factor,
                                                                       alpha=q_alpha,
                                                                       epsilon=q_epsilon,
                                                                       double=False,
                                                                       seed=i)
        results["Q"]["episode_length"].append(np.array(episode_lengths))
        results["Q"]["episode_return"].append(np.array(episode_returns))
        results["Q"]["q_values"].append(Q_values[:, :, 0])

        Q_values, (episode_lengths, episode_returns) = run_experiment( env,
                                                                       num_episodes,
                                                                       discount_factor=dq_discount_factor,
                                                                       alpha=dq_alpha,
                                                                       epsilon=dq_epsilon,
                                                                       double=True,
                                                                       seed=i)
        results["DQ"]["episode_length"].append(np.array(episode_lengths))
        results["DQ"]["episode_return"].append(np.array(episode_returns))
        results["DQ"]["q_values"].append(Q_values.mean(-1))
    
    for i in results.keys():
        for j in results[i].keys():
            array = np.array(results[i][j])
            results[i][j] = [array.mean(0), array.std(0)]
            
    return results