import gym
import numpy as np
from model import DQNModel
from policy import EpsGreedyPolicy
from memory import Memory
from agent import DQNAgent
from processor import AtariProcessor

if __name__ == '__main__':

    ENV_NAME = 'Riverraid-v4'
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    model = DQNModel(nb_actions=nb_actions).model
    policy = EpsGreedyPolicy(eps_min=0.1, eps_max=1, eps_test=0.05, nb_steps=1000000)
    memory = Memory(max_len=1000000)
    processor = AtariProcessor()
    dqn = DQNAgent(env, model, policy, memory, processor, gamma=0.99, batch_size=32,
                   target_model_update_steps=10000, nb_episodes_warmup=500)

    dqn.fit(nb_episodes=20000, action_repetition=1, save_weights=True, save_weights_step=1000, weights_folder='./', visualize=True)

    # file = './weights.h5f'
    # dqn.load_weights(file)
    dqn.test(nb_episodes=10, visualize=True)
