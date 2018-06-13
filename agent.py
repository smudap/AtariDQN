from keras.models import clone_model
import numpy as np
import os
import time

class DQNAgent:
    def __init__(self, env, model, policy, memory, processor, gamma=0.99, batch_size=32,
                target_model_update_steps = 10000, nb_episodes_warmup = 75):
        self.step = 0
        self.episode = 0
        self.env = env
        self.nb_actions = env.action_space.n
        self.model = model
        target_model = clone_model(model)
        target_model.set_weights(model.get_weights())
        self.target_model = target_model
        self.policy = policy
        self.memory = memory
        self.processor = processor
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_model_update_steps = target_model_update_steps
        self.nb_episodes_warmup = nb_episodes_warmup
        self.training = False
        
    def load_weights(self, weights_filename):
        self.model.load_weights(weights_filename)
        self.target_model.load_weights(weights_filename)
        
    def save_weights(self, weights_filename, overwrite=False):
        self.model.save_weights(weights_filename, overwrite=overwrite)
    
    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        
    def process_state_batch(self, batch):
        batch = np.array(batch)
        return self.processor.process_state_batch(batch)
    
    def compute_q_values(self, state_batch):
        state_batch = np.array([state_batch])
        state_batch = self.process_state_batch(state_batch)
        q_values = self.model.predict_on_batch(state_batch).flatten()
        return q_values        
    
    def take_action(self, observation):
        states = self.memory.get_recent_states(observation)
        q_values = self.compute_q_values(states)
        action = self.policy.select_action(q_values, self.step, self.training)
        self.recent_observation = observation
        self.recent_action = action
        return action
    
    def learn_from_action(self, observation, reward, terminal):
        self.memory.append(self.recent_observation, self.recent_action, reward, terminal, observation)
        if self.episode > self.nb_episodes_warmup and self.training:
            experiences = self.memory.sample_states(self.batch_size)
            observations_batch = []
            actions_batch = []
            rewards_batch = []
            terminals_batch = []
            observation_after_actions_batch = []
            for experience in experiences:
                observations_batch.append(experience[0])
                actions_batch.append(experience[1])
                rewards_batch.append(experience[2])
                terminals_batch.append(experience[3])
                observation_after_actions_batch.append(experience[4])
            observations_batch = self.process_state_batch(observations_batch)
            observation_after_actions_batch = self.process_state_batch(observation_after_actions_batch)
            actions_batch = np.array(actions_batch)
            rewards_batch = np.array(rewards_batch)
            terminals_batch = np.array(terminals_batch)
            target_q_values = self.target_model.predict_on_batch(observation_after_actions_batch)
            q_max_batch = np.max(target_q_values, axis=1).flatten()
            discounted_rewards_batch = self.gamma * q_max_batch * (~terminals_batch).astype('float')
            y = rewards_batch + discounted_rewards_batch
            Y = np.zeros((self.batch_size, self.nb_actions))
            for i in range(self.batch_size):
                Y[i][actions_batch[i]] = y[i]
            metric_mse = self.model.train_on_batch(observations_batch, Y)
            if self.step % self.target_model_update_steps == 0:
                self.target_model.set_weights(self.model.get_weights())
            return metric_mse
        else:
            return 0
        
    def fit(self, nb_episodes, action_repetition=1, save_weights=True, save_weights_step=1000, weights_folder=None,
            visualize=False, visualize_sleep=0.02):
        self.training = True
        self.step = 0
        self.episode = 0
        if save_weights and weights_folder is None:
            weights_folder = '.'
        while self.episode < nb_episodes:
            episode_start_time = time.time()
            episode_step = 0
            episode_reward = 0
            self.episode += 1
            self.reset_states()
            observation = self.env.reset()
            observation = self.processor.process_observation(observation)
            
            terminal_state = False
            while not terminal_state:
                self.step += 1
                episode_step += 1
                action = self.take_action(observation)
                reward = 0
                for _ in range(action_repetition):
                    observation, r, terminal_state, info = self.env.step(action)
                    observation, r, terminal_state, info = self.processor.process_step(observation, r,
                                                                                       terminal_state, info)
                    reward += r
                if visualize:
                    time.sleep(visualize_sleep)
                    self.env.render(mode='human')
                self.learn_from_action(observation, reward, terminal_state)
                episode_reward += reward
            
            self.take_action(observation)
            metric_mse = self.learn_from_action(observation, 0, terminal_state)
            episode_duration = time.time() - episode_start_time
            
            print('{}, Step {}, Episode {}: duration {:.3f}, steps {}, eps {:.2f}, reward per episode {}, reward per step {:.3f}, mse per episode {:.3f}'.\
                  format(time.asctime(),
                         self.step,
                         self.episode,
                         episode_duration,
                         episode_step,
                         self.policy.get_actual_eps(self.step, self.training),
                         episode_reward,
                         episode_reward/episode_step,
                         metric_mse
                         ))
            if save_weights and self.episode % save_weights_step == 0:
                weights_file = '{}_weights_{}.h5f'.format(self.env.spec.id, self.episode)
                self.save_weights(os.path.join(weights_folder, weights_file), overwrite=True)
        if visualize:
            self.env.close()
        if save_weights:
            weights_file = '{}_weights_{}.h5f'.format(self.env.spec.id, self.episode)
            self.save_weights(os.path.join(weights_folder, weights_file), overwrite=True)
            
            
    def test(self, nb_episodes, action_repetition=1, visualize=False, visualize_sleep=0.02):
        self.training = False
        self.step = 0
        self.episode = 0
        while self.episode < nb_episodes:
            episode_start_time = time.time()
            episode_step = 0
            episode_reward = 0
            self.episode += 1
            self.reset_states()
            observation = self.env.reset()
            observation = self.processor.process_observation(observation)
            
            terminal_state = False
            while not terminal_state:
                self.step += 1
                episode_step += 1
                action = self.take_action(observation)
                reward = 0
                for _ in range(action_repetition):
                    observation, r, terminal_state, info = self.env.step(action)
                    observation, r, terminal_state, info = self.processor.process_step(observation, r,
                                                                                       terminal_state, info)
                    reward += r
                if visualize:
                    time.sleep(visualize_sleep)
                    self.env.render(mode='human')
                self.learn_from_action(observation, reward, terminal_state)
                episode_reward += reward
            
            self.take_action(observation)
            self.learn_from_action(observation, 0, terminal_state)
            episode_duration = time.time() - episode_start_time
            
            print('{}, Step {}, Episode {}: duration {:.3f}, steps {}, eps {:.2f}, reward per episode {}, reward per step {:.3f}'.\
                  format(time.asctime(),
                         self.step,
                         self.episode,
                         episode_duration,
                         episode_step,
                         self.policy.get_actual_eps(self.step, self.training),
                         episode_reward,
                         episode_reward/episode_step
                         ))
        if visualize:
            self.env.close()
