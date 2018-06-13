import numpy as np

class EpsGreedyPolicy:
    def __init__(self, eps_min=0.1, eps_max=1, eps_test=0.5, nb_steps=1000000):
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_test = eps_test
        self.nb_steps = nb_steps
    
    def get_actual_eps(self, actual_step, training):
        if training:
            # Linear annealed: f(x) = b - ax.
            a = (self.eps_max - self.eps_min) / self.nb_steps
            return max(self.eps_max - a * actual_step, self.eps_min)
        else:
            return self.eps_test
    
    def select_action(self, Q_values, actual_step, training):
        eps = self.get_actual_eps(actual_step, training)
        nb_actions = len(Q_values)
        if np.random.uniform() < eps:
            action = np.random.randint(0, nb_actions - 1)
        else:
            action = np.argmax(Q_values)
        return action
