import numpy as np





class FatigueMDP2():

    def __init__(self):

        self.num_fatigue_states = 10
        self.num_actions = 21
        self.start_state = 0
        self.w_0  = 17
        self.mu = 1
        self.lamda = 1
        self.num_bins_fatigue = 10
        self.discrete_bins = np.linspace(0, 1, self.num_bins_fatigue+1)
        

    def next_state(self, current_state, taskload):

       

        R_t = current_state * np.exp(-self.mu * max((self.w_0 - taskload), 0))

        F_next = R_t + (1 - R_t) * (1 - np.exp(-self.lamda * taskload))

        # we need to discretize the taskload here 
        
       
        discrete_bins = np.digitize(F_next, self.discrete_bins)
       
        next_state = F_next
        return next_state, discrete_bins
    
    