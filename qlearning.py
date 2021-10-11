class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon, double=False):
        self.Q = np.stack([Q, Q], axis=-1)
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        Q = np.sum(self.Q, axis=-1) 
        action_probs = Q[obs]
        if np.random.uniform() < 1-self.epsilon:
            action = np.argmax(action_probs)
        else:
            action = np.random.choice(np.arange(len(self.Q[obs])))
        return action
    
    
def update(Q, state, state_prime, action, reward, discount_factor, alpha, double=False):
    if double:
        index = np.random.choice([0, 1])
    else:
        index = 0
    Q[state, action, index] = Q[state, action, index] + alpha*(reward + discount_factor*max(Q[state_prime, :, index]) - Q[state, action, index])
    return Q
    

def q_learning(env, num_episodes, policy_fn, update_fn, discount_factor=1.0, alpha=0.5, epsilon=0.1, double=False):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        policy: A behavior policy which allows us to sample actions with its sample_action method.
        Q: Q value function
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
    policy = policy_fn(np.zeros((env.nS, env.nA)), epsilon=epsilon)
    
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        
        state = env.reset()
        
        while True:
            action = policy.sample_action(state)
            state_prime, reward, termination, info = env.step(action)
            
            policy.Q = update_fn(policy.Q, state, state_prime, action, reward, discount_factor, alpha, double=double)
            
            state = state_prime
            
            i += 1
            R += reward
            
            if termination:
                break
        
        stats.append((i, R))
    episode_lengths, episode_returns = zip(*stats)
    return policy.Q, (episode_lengths, episode_returns)
