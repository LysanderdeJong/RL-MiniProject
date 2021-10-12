import gym

def load_environment(environment):

    if environment == 'FrozenLake-v1-nonslippery':

        env = gym.make('FrozenLake-v1', is_slippery=False)
        print(env.desc)

    else:
        env = gym.make(environment)

    return env