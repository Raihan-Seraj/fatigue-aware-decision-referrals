import numpy as np





class FatigueMDP():

    def __init__(self):

        self.num_fatigue_states = 4
        self.num_actions = 4
        self.start_state_probs = np.array([1,0,0,0])#np.ones(self.num_fatigue_states)/(self.num_fatigue_states)
        




        # self.action_transition_matrix = {0: np.array([[0.5, 0.3,0.1,0.1],
        #                                               [0.4, 0.3, 0.2, 0.1],
        #                                               [0.3,0.4,0.2,0.1],
        #                                               0.3,0.3,0.2,0.2]), 
                                                      
        #                                  1: np.array([[]])            }

        self.P = {0: np.array([[0.7,0.3,0,0],
                                [0.5, 0.4, 0.1, 0],
                                [0.3, 0.4, 0.3, 0],
                                [0.3, 0.4, 0.2, 0.1]]),
                                
                 1: np.array([[0.4, 0.5, 0.1, 0],
                                [0.2, 0.4, 0.3, 0.1],
                                [0.1, 0.2, 0.4, 0.3],
                                [0,0.1,0.2,0.7]]),
                                
                 2: np.array([[0.1,0.4,0.4,0.1],
                            [0.1,0.2,0.4,0.3],
                            [0,0.1,0.3,0.6],
                            [0,0,0.2,0.8]]),
                 
                 3: np.array([[0.1, 0.1, 0.4, 0.4],
                            [0,0.1,0.3,0.6],
                            [0,0,0.2,0.8],
                            [0,0,0.1,0.9]])
                            
                            }

    def next_state(self, current_state, taskload_discretized):

        # we need to discretize the taskload here 
        
        assert isinstance(taskload_discretized, np.int64), "Value is not an integer"
        assert 0 <= taskload_discretized <= 3, "Accepted action cannot be outside of [0,3]"
    

        next_state = np.random.choice(self.num_fatigue_states, p=self.P[taskload_discretized][current_state,:])

        return next_state
    
    