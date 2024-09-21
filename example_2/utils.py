import numpy as np 
import tqdm 
from scipy.stats import norm





##########################################################################################################################
#
## All the mathematical expresssions in this file can be obtained from https://hackmd.io/@r6RYMGOsSsmik_7-7ToO3Q/rJnCXWQqR
#
###########################################################################################################################





class Utils(object):
    '''
    The class is initialized with the following arguments
    
    Args:
        num_tasks_per_batch: An integer denoting the total number of tasks in a given batch.

        mu: The fatigue recovery rate --> input type float 

        lamda: The fatigue growth rate --> input type float

        w_0: The level of fatigue beyond which fatigue recovery does not occur --> input type integer

        sigma_a: The observation variance of the gaussian distribution for the automation --> input any positive real

        H0: The value of the null hypothesis --> input type integer

        H1: The value of the alternate hypothesis --> input type integer  

        prior: The prior distribution of H0 and H1 --> input type in the form [P(H0), P(H1)] where P(H0) and P(H1) denotes the probability of H0 and H1 respectively

        d_0: The scaling term for the mean of the gaussian distribution for the observation of the human.

        beta: The extent of the influence of fatigue and taskload on the observation --> input type value between 0 and 1 inclusive.

        sigma_h: The observation variane of the gaussian distribution of the human --> input type: any positive real value

        ctp: True positive costs --> input type any positive real value

        ctn: True negative costs --> input type any positive real value

        cfp: False positive cost ---> input type any positive real value
        
        cfn: False negative costs --> input type any postive real value

        num_bins_fatigue: The number of bins used to discretize the fatigue state --> input type int 

        


    
    '''


    def __init__(self,num_tasks_per_batch, mu, lamda,w_0,sigma_a,H0,H1,prior,d_0,alpha,beta,sigma_h,ctp,ctn,cfp,cfn,cm,num_bins_fatigue):

        self.num_tasks_per_batch = num_tasks_per_batch
        self.mu  = mu
        self.lamda = lamda 
        self.w_0 = w_0
        self.sigma_a = sigma_a
        self.H0 = H0
        self.H1 = H1
        self.prior = prior 
        self.d_0 = d_0
        self.alpha = alpha
        self.beta = beta
        self.sigma_h = sigma_h
        self.ctp = ctp 
        self.ctn = ctn 
        self.cfp = cfp 
        self.cfn = cfn
        self.cm = cm
        self.num_bins_fatigue = num_bins_fatigue 

        pass


    
    '''
    Function that provides the next fatigue state given the current fatigue state and workload 

    Args:
        F_t: The current fatigue state in the range [0,1]
        w_t: The taskload provided to the human 


    '''

    def get_fatigue(self,F_t, w_t):

        R_t = F_t * np.exp(-self.mu * max((self.w_0 - w_t), 0))

        F_next = R_t + (1 - R_t) * (1 - np.exp(-self.lamda * w_t))

        return F_next



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


    '''
    A function that computes the value of the threshold tau for a given workload and fatigue state
    
    Args:
        F_t: The level of fatigue in the range [0,1] --> input type values between 0 and 1 inclusive 
    
        w_t: The taskload --> input type integer
    '''
    ## taking max at the denominator for stability 
    def tau(self,w_t, F_t):

        tau = self.d_0 * (1 - (1-np.exp(-self.alpha*F_t)*(min(w_t/self.w_0,1))**self.beta)) / 2 + (
            self.sigma_h**2 / max(self.d_0 * (1 - (1-np.exp(-self.alpha*F_t)*(min(w_t/self.w_0,1))**self.beta)),1e-20)
        ) * np.log((self.cfp - self.ctn) * self.prior[0] / ((self.cfn - self.ctp) * self.prior[1]))

        return tau


    ###############################################################################################################################



    '''
    Function that computes the value of gamma

    Args:
        automation posterior: A list in the form [P(H0|Y), P(H1|Y)] for the automation observation 
        
        w_t: The taskload at time t --> input type integer.
        
        F_t: The level of fatigue in the range [0,1] --> input type values between 0 and 1 inclusive 
    '''


    def compute_gamma(self,automation_posterior, w_t, F_t):

        tau_wf = self.tau(w_t, F_t)

        # computing true positive probability of the human
        P_h_tp = 1 - norm.cdf(
            (tau_wf - self.d_0 * (1 - (1-np.exp(-self.alpha*F_t)*(min(w_t/self.w_0,1))**self.beta))) / self.sigma_h,
            loc=0,
            scale=1,
        )

        # computing false positive probability of the human
        P_h_fp = 1 - norm.cdf((tau_wf) / self.sigma_h, loc=0, scale=1)

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

    def compute_G(self,automation_posterior, w_t, F_t):

        # automation_posterior is a list where [p(h0|y), p(h1|y)]
        # C_a = min(posterior_h0_k* cost(h_0,0)+posterior_h1_k*cost(h_0,1), posterior_h0_k* cost(h_1,0)+posterior_h1_k*cost(h_1,1))

        # the following expression is not originally in the paper but in the formulation write up and it is the same as the original cdc paper
        C_a = min(
            self.ctn + (self.cfn - self.ctn) * automation_posterior[1],
            self.cfp + (self.ctp - self.cfp) * automation_posterior[1],
        )

        # get the value of gamma_bar_2

        gamma = self.compute_gamma(
            automation_posterior, w_t, F_t
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

        total_indices = list(range(self.num_tasks_per_batch))#[i for i in range(self.num_tasks_per_batch)]

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

        tau_wf = self.tau(w_t, F_t)

        # computing true positive probability of the human

        P_h_tp = 1 - norm.cdf(
            (tau_wf - self.d_0 * (1 - (1-np.exp(-self.alpha*F_t)*(min(w_t/self.w_0,1))**self.beta))) / self.sigma_h,
            loc=0,
            scale=1,
        )

        # computing false positive probability of the human

        P_h_fp = 1 - norm.cdf((tau_wf) / self.sigma_h, loc=0, scale=1)

        human_cost_all_deferred_tasks = [
            (1 - batched_posterior_h1[i]) * (P_h_fp * self.cfp + (1 - P_h_fp) * self.ctn)
            + batched_posterior_h1[i] * (P_h_tp * self.ctp + (1 - P_h_tp) * self.cfn)
            for i in deferred_task_indices
        ]

        human_cost_per_batch = sum(human_cost_all_deferred_tasks)
        total_per_step_cost = auto_cost_per_batch + deferred_cost + human_cost_per_batch

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
                w_t,
                F_t,
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
                    w_t,
                    F_t,
                    
                )
                for k in human_allocation_indices
            ]
        )

        return human_allocation_indices, G_bar_w
    



    ################################################################################################################################

    '''
    Function that discretize the fatigue states 

    Args:
        F_t: The fatigue state at time t in the range [0,1]
    '''
    def discretize_fatigue_state(self,F):

        bins = np.linspace(0, 1, self.num_bins_fatigue + 1)

        discretized_data = np.digitize(F, bins)

        # subtracting 1 since python is zero indexed
        return discretized_data - 1
    

    ###################################################################################################################################


   
    def compute_kesav_policy(self,F_t, batched_posterior_h0, batched_posterior_h1):

        total_num_tasks = len(batched_posterior_h0)

        all_gbar_w = []
        all_human_indices = []
        for w_t in range(total_num_tasks+1):

            human_allocation_indices, G_bar_w = self.algorithm_1_per_wl(batched_posterior_h0,batched_posterior_h1,w_t,F_t)

            all_gbar_w.append(G_bar_w)
            all_human_indices.append(human_allocation_indices)
        
        max_idx = np.argmax(all_gbar_w)

        w_t_star = max_idx 

        final_human_indices = all_human_indices[max_idx]

        return w_t_star, final_human_indices



