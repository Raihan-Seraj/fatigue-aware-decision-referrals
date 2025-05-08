import numpy as np

class FatigueMDP2():

    def __init__(self):

        self.num_fatigue_states = 11
        self.num_actions = 21
        self.start_state = 0
        self.num_bins_fatigue = 10
        self.discrete_bins = np.linspace(0, 1, self.num_bins_fatigue)
        self.fatigue_states = [0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        self.rho = 0.001
       


    def next_state(self, current_state, taskload):

       
        F_next = min(current_state + self.rho * taskload**2 ,1)

        
       
        F_next_discretized = np.digitize(F_next, self.discrete_bins)
       
        next_state_discretized = F_next_discretized-1
        next_state_raw = F_next
        return next_state_raw, next_state_discretized
    
    
