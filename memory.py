import numpy as np
from collections import deque

class Memory:
    def __init__(self, max_len=1000000, input_size=4):
        self.input_size = input_size
        self.recent_observations = deque(maxlen=input_size)
        self.recent_terminals = deque(maxlen=input_size)
        self.observations = deque(maxlen=max_len)
        self.observations_after_action = deque(maxlen=max_len)
        self.terminals = deque(maxlen=max_len)
        self.actions = deque(maxlen=max_len)
        self.rewards = deque(maxlen=max_len)
        self.nb_observations = 0
    
    def empty_observation(self, observation):
        return np.zeros(observation.shape)
    
    def get_recent_states(self, observation):
        states = [observation]
        for i in reversed(range(1, len(self.recent_observations))):
            terminal = self.recent_terminals[i - 1] if i >= 1 else False
            if terminal:
                break
            states.insert(0, self.recent_observations[i])
        while len(states) < self.input_size:
            states.insert(0, self.empty_observation(states[0]))
        return states
        
    def append(self, observation, action, reward, terminal, observation_after_action):
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)
        self.observations.append(observation)
        self.terminals.append(terminal)
        self.observations_after_action.append(observation_after_action) \
            if terminal == False else self.observations_after_action.append(self.empty_observation(observation))
        self.actions.append(action)
        self.rewards.append(reward)
        self.nb_observations += 1
        
    def sample_states(self, batch_size):
        start_indexes = np.random.choice(range(self.nb_observations), batch_size)
        experiences = []
        for i in start_indexes:
            states = []
            for j in range(0, self.input_size):
                k = i - j
                terminal = self.terminals[k - 1] if k >= 1 else True
                if terminal:
                    break
                states.insert(0, self.observations[k])
            while len(states) < self.input_size:
                states.insert(0, self.empty_observation(self.observations[0]))
            action = self.actions[i]
            reward = self.rewards[i]
            states_after_actions = [np.copy(x) for x in states[1:]]
            states_after_actions.append(self.empty_observation(self.observations[0])) if self.terminals[i-1] else \
                states_after_actions.append(self.observations_after_action[i])
            terminal = self.terminals[i]
            experiences.append([states, action, reward, terminal, states_after_actions])
        return experiences
