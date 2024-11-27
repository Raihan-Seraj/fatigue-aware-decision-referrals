import numpy as np 
from scipy.stats import norm
import scipy.special as sp
import scipy.stats as stats
from envs.fatigue_model_1 import FatigueMDP
from envs.fatigue_model_2 import FatigueMDP2
import matplotlib
#matplotlib.use('Agg')






##########################################################################################################################
#
## All the mathematical expresssions in this file can be obtained from https://hackmd.io/@r6RYMGOsSsmik_7-7ToO3Q/rJnCXWQqR
#
###########################################################################################################################





class Utils(object):
    '''
    The class is initialized with the argparse 
    
    '''


    def __init__(self, args):

        self.args = args
        self.num_tasks_per_batch = self.args.num_tasks_per_batch
        self.sigma_a = self.args.sigma_a
        self.H0 = self.args.H0
        self.H1 = self.args.H1    
        self.prior = self.args.prior 
        self.d_0 = self.args.d_0
        self.alpha = self.args.alpha
        self.beta = self.args.beta
        self.gamma = self.args.gamma
        self.ctp = self.args.ctp 
        self.ctn = self.args.ctn 
        self.cfp = self.args.cfp 
        self.cfn = self.args.cfn
        self.cm = self.args.cm
        self.cfr = np.log( ( (self.cfp-self.ctn)*self.prior[0])/ ((self.cfn-self.ctp)*self.prior[1])  )
        self.model_name = args.model_name
        
        if args.model_name.lower()=='fatigue_model_1':
            self.env = FatigueMDP()
            # number of bins to discretize the taskload 
            self.num_bins = np.linspace(0, self.num_tasks_per_batch, self.env.num_actions)
        elif args.model_name.lower()=='fatigue_model_2':
            self.env = FatigueMDP2()
        
        else:
            raise ValueError("Invalid Fatigue Model")
        

        

    
    def discretize_taskload(self,w_t):

        discretized_data = np.digitize(w_t, self.num_bins)

        # subtracting 1 since python is zero indexed
        return discretized_data - 1



    


    #####################################################################################################################


    '''
    Function that provides a batch of observation corresponding to the batch of tasks

    Returns:
        batched_obs: A list consisting of observations associated with each tasks.
        batched_posterior_h0: A list consisting of the posteriror value of H0 given observation. 
        batched_posterior_h1: A list consisting of posterior value of H1 given observation. 
    '''

    def get_auto_obs(self):

        # we consider H0=0,  and H1=1

        batched_obs = []
        batched_posterior_h0 = []
        batched_posterior_h1 = []

        for i in range(self.num_tasks_per_batch):

            task_state = np.random.choice([self.H0, self.H1], p=self.prior)

            obs_automation = task_state + np.random.normal(0, self.sigma_a)

            batched_obs.append(obs_automation)

            # computing posterior probability

            liklihood_h0 = norm.pdf(obs_automation, loc=self.H0, scale=self.sigma_a)
            liklihood_h1 = norm.pdf(obs_automation, loc=self.H1, scale=self.sigma_a)

            posterior_h0 = (
                self.prior[0]
                * liklihood_h0
                / (liklihood_h0 * self.prior[0] + liklihood_h1 * self.prior[1])
            )

            posterior_h1 = 1 - posterior_h0

            batched_posterior_h0.append(posterior_h0)
            batched_posterior_h1.append(posterior_h1)

        return batched_obs, batched_posterior_h0, batched_posterior_h1
    

    #########################################################################################################################

    def Phfp(self, F_t, w_t):

        if self.model_name.lower()=='fatigue_model_1':

            # false_pos_dict = {0:{0: 0.01, 1: 0.02, 2: 0.03},
            #                     1:{0: 0.05, 1: 0.07, 2: 0.09},
            #                     2:{0:0.3, 1:0.34, 2:0.35},
            #                     3:{0:0.6, 1: 0.66, 2: 0.7},
            #                     4:{0:0.9, 1:0.95, 2: 0.99}}
            w_t_d = self.discretize_taskload(w_t)
    
            false_pos_func = (0.3 * F_t + 0.1 * w_t_d)/3
            # res = false_pos_dict[F_t][w_t_d]


            res = false_pos_func
            assert 0 <= res <= 1, "The probability of false positive should be [0,1]"
        
        elif self.model_name.lower()=='fatigue_model_2':

            res = min(F_t + 0.001 *w_t,1)
        
        else:
            raise ValueError("Invalid fatigue model")

       

        return res
    
    def Phtp(self, F_t, w_t):
        

        if self.model_name.lower()=='fatigue_model_1':
        

            w_t_d = self.discretize_taskload(w_t)
            true_pos_func = 1 - (0.2*F_t + 0.1*w_t_d)/2.5

            res_1 = true_pos_func
            
            #res_1 = 1-F_t/5 - 0.2 *w_t/20

            assert 0 <= res_1 <= 1, "The probability of true positive should be [0,1]"

            res = res_1
            #res = res_1
        
        elif self.model_name.lower()=='fatigue_model_2':
            res_1 = self.Phfp(F_t, w_t)
            res = max(res_1,1e-8) ** self.gamma
        return res




    ##########################################################################################################################


    '''
    Function that computes the value of gamma

    Args:
        automation posterior: A list in the form [P(H0|Y), P(H1|Y)] for the automation observation 
        
        w_t: The taskload at time t --> input type integer 
        
        F_t: The level of fatigue in the range [0,1] --> input type values between 0 and 1 inclusive 
    '''


    def compute_gamma(self,automation_posterior, F_t, w_t):
        
        #computing false positive probability
        P_h_fp = self.Phfp(F_t, w_t)
        
        # computing true positive probability of the human
        P_h_tp = self.Phtp(F_t,w_t)

       
        gamma = automation_posterior[1] * (
            P_h_tp * self.ctp + (1 - P_h_tp) * self.cfn
        ) + automation_posterior[0] * (P_h_fp * self.cfp + (1 - P_h_fp) * self.ctn)

        
        return gamma
    

    #############################################################################################################################################

    '''
    Function that computes the value of G (see the notes) for given automation posterior taskload and fatigue level
    
    Args:
        automation posterior: A list in the form [P(H0|Y), P(H1|Y)] for the automation observation 
        
        w_t: The taskload at time t --> input type integer.
        
        F_t: The level of fatigue in the range [0,1] --> input type values between 0 and 1 inclusive 
    '''

    def compute_G(self,automation_posterior, F_t, w_t):

        # automation_posterior is a list where [p(h0|y), p(h1|y)]
        # C_a = min(posterior_h0_k* cost(h_0,0)+posterior_h1_k*cost(h_0,1), posterior_h0_k* cost(h_1,0)+posterior_h1_k*cost(h_1,1))

        # the following expression is not originally in the paper but in the formulation write up and it is the same as the original cdc paper
        C_a = min(
            self.ctn + (self.cfn - self.ctn) * automation_posterior[1],
            self.cfp + (self.ctp - self.cfp) * automation_posterior[1],
        )

        # get the value of gamma_bar_2

        gamma = self.compute_gamma(
            automation_posterior, F_t,w_t
        )

       

        G = C_a - gamma - self.cm

        return G
    
    #############################################################################################################################################


    '''
    Function that computes the perstep cost for a given batch

    Args:
        F_t: The level of fatigue in the range [0,1] --> input type values between 0 and 1 inclusive 
        

        batched_posterior_h1: A list consisting of posterior value of H1 given observation. 

        deferred_indices: A list of task indices that are deferred to the human



    '''


    def per_step_cost(self,F_t, batched_posterior_h1, deferred_task_indices):

        total_indices = list(range(self.num_tasks_per_batch))

        auto_indices = list(set(total_indices)-set(deferred_task_indices))
       
        w_t = len(deferred_task_indices)

        auto_cost_all_tasks = [
            min(
                self.ctn + (self.cfn - self.ctn) * batched_posterior_h1[i],
                self.cfp + (self.ctp - self.cfp) * batched_posterior_h1[i],
            )
            for i in auto_indices
        ]

        auto_cost_per_batch = sum(auto_cost_all_tasks)
        deferred_cost = len(deferred_task_indices) * self.cm

       
        
        #computing false positive probability
        P_h_fp = self.Phfp(F_t,w_t)
        
        # computing true positive probability of the human
        P_h_tp = self.Phtp(F_t,w_t)
        

        human_cost_all_deferred_tasks = [
            (1 - batched_posterior_h1[i]) * (P_h_fp * self.cfp + (1 - P_h_fp) * self.ctn)
            + batched_posterior_h1[i] * (P_h_tp * self.ctp + (1 - P_h_tp) * self.cfn)
            for i in deferred_task_indices
        ]

        human_cost_per_batch = sum(human_cost_all_deferred_tasks)
        #total_per_step_cost = auto_cost_per_batch + deferred_cost + human_cost_per_batch

        return auto_cost_per_batch, human_cost_per_batch, deferred_cost


    #####################################################################################################################################



    '''
    Function that computes the value of c* (see the notes) at the top of this file

    Args:
        w_t: The taskload at time t --> input type integer.
        
        F_t: The level of fatigue in the range [0,1] --> input type values between 0 and 1 inclusive. 

        batched_posterior_h0: A list consisting of posterior value of H0 given observation. 


        batched_posterior_h1: A list consisting of posterior value of H1 given observation. 

        
    '''

    def compute_cstar(self, F_t, w_t, batched_posterior_h0, batched_posterior_h1):

        sum_C_a = sum(
            [
                min(
                    self.ctn + (self.cfn - self.ctn) * batched_posterior_h1[i],
                    self.cfp + (self.ctp - self.cfp) * batched_posterior_h1[i],
                )
                for i in range(self.num_tasks_per_batch)
            ]
        )

        deferred_indices, sum_G = self.algorithm_1_per_wl(batched_posterior_h0, batched_posterior_h1, w_t, F_t)
         

        c_star = sum_C_a - sum_G

        return c_star, deferred_indices, sum_G
    
    ########################################################################################################################################

    '''
    Function that computes kesav's algorithm 
    
    Args:

        batched_posterior_h0: A list consisting of posterior value of H0 given observation. 
        batched_posterior_h1: A list consisting of posterior value of H1 given observation. 

        w_t: The taskload at time t --> input type integer.
        
        F_t: The level of fatigue in the range [0,1] --> input type values between 0 and 1 inclusive. 
    
    '''



    def algorithm_1_per_wl(self, batched_posterior_h0, batched_posterior_h1, w_t,F_t ):
    

        num_tasks = self.num_tasks_per_batch

        G_vals = [
            self.compute_G(
                [batched_posterior_h0[k], batched_posterior_h1[k]],
                F_t,
                w_t,
            )
            for k in range(num_tasks)
        ]

        # get the indices when the g vals are sorted in descending order
        sorted_indices = [
            index
            for index, value in sorted(enumerate(G_vals), key=lambda x: x[1], reverse=True)
        ]

        ## allocate the first wK task to the human

        wk = w_t

        human_allocation_indices = sorted_indices[:wk]

        # computing the cost reduction
        G_bar_w = sum(
            [
                self.compute_G(
                    [batched_posterior_h0[k], batched_posterior_h1[k]],
                    
                    F_t,
                    w_t
                    
                )
                for k in human_allocation_indices
            ]
        )

        return human_allocation_indices, G_bar_w
    



    ################################################################################################################################

    
   
    def compute_kesav_policy(self,F_t, batched_posterior_h0, batched_posterior_h1):

        

        all_gbar_w = []
        all_human_indices = []
        for w_t in range(self.num_tasks_per_batch):

            human_allocation_indices, G_bar_w = self.algorithm_1_per_wl(batched_posterior_h0,batched_posterior_h1,w_t,F_t)

            all_gbar_w.append(G_bar_w)
            all_human_indices.append(human_allocation_indices)
        
        max_idx = np.argmax(all_gbar_w)

        w_t_star = max_idx 

        final_human_indices = all_human_indices[max_idx]

        return w_t_star, final_human_indices


    ## functions for computing inverse and q func
    def qfunc(self,x):
        return 0.5 * sp.erfc(x / np.sqrt(2))
    
    def qfuncinv(self, y):
        return np.sqrt(2) * stats.norm.ppf(1 - y)


    