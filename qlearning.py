import numpy as np
from tqdm import tqdm

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon, decay_rate, double=False):
        self.Q = np.stack([Q, Q], axis=-1)
        self.epsilon = epsilon
        self.double = double
        self.decay_rate = decay_rate
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        if self.double:
            Q = np.mean(self.Q, axis=-1)
        else:
            Q = self.Q[..., 0]
            
        if isinstance(obs, tuple):
            obs = tuple((int(d) for d in obs))
            
        action_probs = Q[obs]
        if np.random.uniform() < 1-self.epsilon:
            action_probs = (action_probs - np.max(action_probs)) == 0
            action_probs = action_probs / action_probs.sum()
            action = np.random.choice(np.arange(len(self.Q[obs])), p=action_probs)
        else:
            action = np.random.choice(np.arange(len(self.Q[obs])))
        return action
    
    def set_epsilon(epsilon):
        self.epsilon = epsilon
    
    def update_Q(self, state, state_prime, action, reward, discount_factor=1.0, alpha=0.5):
        if self.double:
            index = 1 if np.random.uniform() > 0.5 else 0
            other_index = 1 - index
        else:
            index = 0
            other_index = 0
        
        if isinstance(state, tuple):
            state = tuple((int(d) for d in state))
            state_prime = tuple((int(d) for d in state_prime))
            Q_index = state + (action, index)
            Q_prime_indices = state_prime + (slice(None), index)
            best_action = np.argmax(self.Q[Q_prime_indices])
            target_Q_index = state_prime + (best_action, other_index)
        else:
            Q_index = (state, action, index)
            Q_prime_indices = (state_prime, slice(None), index)
            best_action = np.argmax(self.Q[Q_prime_indices])
            target_Q_index = (state_prime, best_action, other_index)
        
        self.Q[Q_index] = self.Q[Q_index] + alpha*(reward + discount_factor*self.Q[target_Q_index] - self.Q[Q_index])
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
    outcomes = []
    
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

        policy.epsilon = policy.epsilon * policy.decay_rate

        if i_episode > (num_episodes-1000):
            outcomes.append(reward)

    success_percentage = np.asarray(outcomes).mean()

    episode_lengths, episode_returns = zip(*stats)
    return Q, (episode_lengths, episode_returns), success_percentage
