from PIL import Image
import numpy as np

class AtariProcessor(object):
    def process_step(self, observation, reward, done, info):
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info

    def process_observation(self, observation, input_shape = (84, 84)):
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        img = img.resize(input_shape).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == input_shape
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

    def process_info(self, info):
        return info

    def process_action(self, action):
        return action
