import numpy as np





class FatigueMDP():

    def __init__(self):

        self.num_fatigue_states = 4
        self.num_actions = 4
        self.fatigue_states = [0,1,2,3]
        


        self.P = {0: np.array([[0.8, 0.2, 0, 0],
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
        

    def construct_transition_matrix(self,num_states, num_actions):
        """
        Constructs a transition probability matrix for an MDP.

        Parameters:
        - num_states (int): Number of states in the MDP.
        - num_actions (int): Number of actions in the MDP.

        Returns:
        - transition_matrix (numpy.ndarray): A 3D array of shape (num_actions, num_states, num_states),
        where transition_matrix[a][s][s'] represents the probability of transitioning from state s
        to state s' under action a.
        """
        if num_states < 2:
            raise ValueError("The number of states must be at least 2.")

        # Initialize the transition matrix with zeros
        transition_matrix = np.zeros((num_actions, num_states, num_states))

        for action in range(num_actions):
            for state in range(num_states):
                if state == num_states - 1:
                    # The last state is absorbing
                    transition_matrix[action, state, state] = 1.0
                else:
                    # Transition probabilities for other states
                    for next_state in range(num_states):
                        if next_state == num_states - 1:
                            # Slightly higher probability for transitioning to the absorbing state
                            transition_matrix[action, state, next_state] = 0.1 + (0.8 * action / (num_actions - 1))
                        elif next_state > state:
                            # Higher probability of transitioning to higher states as actions increase
                            transition_matrix[action, state, next_state] = 0.8 * (action + 1) / num_actions
                        else:
                            # Lower probability for self-transition and transitioning to lower states
                            transition_matrix[action, state, next_state] = 0.2 * (1 - (action / num_actions))

                    # Normalize to ensure probabilities sum to 1
                    transition_matrix[action, state, :] /= np.sum(transition_matrix[action, state, :])

        return transition_matrix

    def next_state(self, current_state, taskload_discretized):
        #print(taskload_discretized)
        # we need to discretize the taskload here 
        
        #assert isinstance(taskload_discretized, np.int64), "Value is not an integer"
        #assert 0 <= taskload_discretized <= 2, "Accepted action cannot be outside of [0,2]"
    

        next_state_discretized = np.random.choice(self.num_fatigue_states, p=self.P[taskload_discretized][current_state,:])
        #next_state_discretized = np.random.choice(self.num_fatigue_states, p=self.P[taskload_discretized,current_state,:])

        next_state_raw = next_state_discretized

        return next_state_raw, next_state_discretized
    
    