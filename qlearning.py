import numpy as np
from tqdm import tqdm

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon, double=False):
        self.Q = np.stack([Q, Q], axis=-1)
        self.epsilon = epsilon
        self.double = double
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        Q = np.sum(self.Q, axis=-1)
        if isinstance(obs, tuple):
            obs = tuple((int(d) for d in obs))
        action_probs = Q[obs]
        if np.random.uniform() < 1-self.epsilon:
            action = np.argmax(action_probs)
        else:
            action = np.random.choice(np.arange(len(self.Q[obs])))
        return action
    
    def set_epsilon(epsilon):
        self.epsilon = epsilon
    
    def update_Q(self, state, state_prime, action, reward, discount_factor=1.0, alpha=0.5):
        if self.double:
            index = 1 if np.random.uniform() > 0.5 else 0
        else:
            index = 0
        
        if isinstance(state, tuple):
            state = tuple((int(d) for d in state))
            state_prime = tuple((int(d) for d in state_prime))
            Q_indices = state + (action, index)
            Q_prime_indices = state + (slice(None), index)
        else:
            Q_indices = (state, action, index)
            Q_prime_indices = (state_prime, slice(None), index)
        
        self.Q[Q_indices] = self.Q[Q_indices] + alpha*(reward + discount_factor*max(self.Q[Q_prime_indices]) - self.Q[Q_indices])
        return self.Q
    

def q_learning(env, num_episodes, policy, discount_factor=1.0, alpha=0.5):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        policy: A behavior policy which allows us to sample actions with its sample_action method.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        
    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """
    
    # Keeps track of useful statistics
    stats = []
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        state = env.reset()
        
        while True:
            action = policy.sample_action(state)
            state_prime, reward, termination, info = env.step(action)
            
            Q = policy.update_Q(state, state_prime, action, reward, discount_factor=discount_factor, alpha=alpha)
            
            state = state_prime
            
            i += 1
            R += reward
            
            if termination:
                break
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns)
