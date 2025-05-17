import numpy as np





class FatigueMDPPerturbed():

    def __init__(self,epsilon=0.2):

        self.num_fatigue_states = 4
        self.num_actions = 4
        self.fatigue_states = [0,1,2,3]
        


        self.P_original = {0: np.array([[0.8, 0.2, 0, 0],
                                [0.3, 0.5, 0.2, 0,],
                                [0, 0.3, 0.5, 0.2],
                                [0, 0, 0 , 1]
                                ]),
                                
                1: np.array([  [0.4, 0.6, 0 , 0],
                                [0.1, 0.3, 0.6, 0],
                                [0, 0.1, 0.3, 0.6],
                                [0, 0, 0, 1]
                                ]),
                                
                2: np.array([[0, 0.7, 0.3, 0],
                              [0, 0, 0.7, 0.3],
                              [0, 0, 0.8,   0.2],
                              [0, 0, 0,    1],
                            ]),
                
                3: np.array([[0, 0.4, 0.6, 0],
                              [0, 0, 0.4, 0.6],
                              [0, 0, 0,   1],
                              [0, 0, 0,    1],
                            ])
                            
                            }
        epsilon=0.05
        
        self.P = {key: np.zeros_like(self.P_original[key]) for key in self.P_original}

        
        for key in self.P:

            self.P[key] = self.perturb_transition_matrix(self.P_original[key],epsilon)
       
     
    
    def next_state(self, current_state, taskload_discretized):
       
    

        next_state_discretized = np.random.choice(self.num_fatigue_states, p=self.P[taskload_discretized][current_state,:])
      
        next_state_raw = next_state_discretized

        return next_state_raw, next_state_discretized

    def perturb_transition_matrix(self, P, epsilon):
        """
        Perturb a transition probability matrix while maintaining valid probability properties.
        
        Args:
            P (np.ndarray): Original transition probability matrix
            epsilon (float): Maximum perturbation magnitude
            
        Returns:
            np.ndarray: Perturbed transition probability matrix
        """
        # Create copy to avoid modifying original
        P_perturbed = P.copy()
        
        # Add random perturbations bounded by epsilon
        perturbations = np.random.uniform(-epsilon, epsilon, size=P.shape)
        P_perturbed += perturbations
        
        # Ensure non-negativity
        P_perturbed = np.maximum(P_perturbed, 0)
        
        # Normalize rows to sum to 1
        row_sums = P_perturbed.sum(axis=1, keepdims=True)
        P_perturbed = P_perturbed / row_sums
        
        return P_perturbed

