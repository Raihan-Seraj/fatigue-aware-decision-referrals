import numpy as np 
from scipy.stats import norm
from envs.fatigue_model_1 import FatigueMDP



class Utils(object):
    """
    Utility class for fatigue-aware decision referrals.
    
    This class provides mathematical functions and algorithms for computing
    optimal task allocation between automation and human operators based on
    fatigue models and task workload.
    
    """

    def __init__(self, args):
        """
        Initialize the Utils class with configuration parameters.
        
        Args:
            args: Argparse object containing configuration parameters
        """
        self.args = args
        self.num_tasks_per_batch = args.num_tasks_per_batch
        self.H0 = args.H0
        self.H1 = args.H1    
        self.prior = args.prior 
        self.ctp = args.ctp 
        self.ctn = args.ctn 
        self.cfp = args.cfp 
        self.cfn = args.cfn
        self.cr = args.cr
        self.sigma_a = args.sigma_a
   
        self.alpha_tp = args.alpha_tp
        self.beta_tp = args.beta_tp
        self.alpha_fp = args.alpha_fp
        self.beta_fp = args.beta_fp
        self.env = FatigueMDP()
        self.num_bins = np.linspace(0, self.num_tasks_per_batch, self.env.num_actions)
       

        
        self.cfr = np.log(((self.cfp - self.ctn) * self.prior[0]) / 
                         ((self.cfn - self.ctp) * self.prior[1]))


    def discretize_taskload(self, w_t):
        """
        Discretize the taskload into bins.
        
        Args:
            w_t (float): The taskload at time t
            
        Returns:
            int: Discretized taskload bin (0-3)
            
        Raises:
            ValueError: If workload is outside valid range
        """
        if 0 <= w_t <= 5:
            return 0
        elif 5 < w_t <= 10:
            return 1
        elif 10 < w_t <= 15:
            return 2
        elif 15 < w_t <= 20:
            return 3
        else:
            raise ValueError(f"Invalid workload: {w_t}. Must be between 0 and 20.")

    def get_auto_obs(self):
        """
        Generate a batch of observations with corresponding posterior probabilities.
        
        Returns:
            tuple: Contains:
                - (batched_obs, ground_truth): Observations and true states
                - batched_posterior_h0: List of P(H0|observation) values
                - batched_posterior_h1: List of P(H1|observation) values
        """
        batched_obs = []
        batched_posterior_h0 = []
        batched_posterior_h1 = []
        ground_truth = []
        
        for _ in range(self.num_tasks_per_batch):
            # Sample true state
            task_state = np.random.choice([self.H0, self.H1], p=self.prior)
            
            # Generate noisy observation
            obs_automation = task_state + np.random.normal(0, self.sigma_a)
            batched_obs.append(obs_automation)
            
            # Compute posterior probabilities using Bayes' rule
            likelihood_h0 = norm.pdf(obs_automation, loc=self.H0, scale=self.sigma_a)
            likelihood_h1 = norm.pdf(obs_automation, loc=self.H1, scale=self.sigma_a)
            
            posterior_h0 = (self.prior[0] * likelihood_h0 / 
                           (likelihood_h0 * self.prior[0] + likelihood_h1 * self.prior[1]))
            posterior_h1 = 1 - posterior_h0
            
            batched_posterior_h0.append(posterior_h0)
            batched_posterior_h1.append(posterior_h1)
            ground_truth.append(task_state)

        return (batched_obs, ground_truth), batched_posterior_h0, batched_posterior_h1

    def Phfp(self, F_t, w_t):
        """
        Compute the false positive probability of the human operator.
        
        Args:
            F_t (int): Fatigue state at time t
            w_t (float): Taskload at time t
            
        Returns:
            float: False positive probability in [0,1]
        """
        w_t_d = self.discretize_taskload(w_t)
        false_pos_prob = min((self.alpha_fp * F_t + self.beta_fp * w_t_d), 1)
        
        assert 0 <= false_pos_prob <= 1, "False positive probability must be in [0,1]"
        return false_pos_prob

    def Phtp(self, F_t, w_t):
        """
        Compute the true positive probability of the human operator.
        
        Args:
            F_t (int): Fatigue state at time t
            w_t (float): Taskload at time t
            
        Returns:
            float: True positive probability in [0,1]
        """
        w_t_d = self.discretize_taskload(w_t)
        true_pos_prob = max(1 - (self.alpha_tp * F_t + self.beta_tp * w_t_d), 0)
        
        assert 0 <= true_pos_prob <= 1, "True positive probability must be in [0,1]"
        return true_pos_prob

    def compute_gamma(self, automation_posterior, F_t, w_t):
        """
        Compute the gamma value for cost calculation.
        
        Args:
            automation_posterior (list): [P(H0|Y), P(H1|Y)] for automation observation
            F_t (int): Fatigue state at time t
            w_t (float): Taskload at time t
            
        Returns:
            float: Gamma value
        """
        P_h_fp = self.Phfp(F_t, w_t)
        P_h_tp = self.Phtp(F_t, w_t)

        gamma = (automation_posterior[1] * (P_h_tp * self.ctp + (1 - P_h_tp) * self.cfn) +
                 automation_posterior[0] * (P_h_fp * self.cfp + (1 - P_h_fp) * self.ctn))
        
        return gamma

    def compute_G(self, automation_posterior, F_t, w_t):
        """
        Compute the G value for task allocation decisions.
        
        Args:
            automation_posterior (list): [P(H0|Y), P(H1|Y)] for automation observation
            F_t (int): Fatigue state at time t
            w_t (float): Taskload at time t
            
        Returns:
            float: G value
        """
        # Automation cost
        C_a = min(
            self.ctn + (self.cfn - self.ctn) * automation_posterior[1],
            self.cfp + (self.ctp - self.cfp) * automation_posterior[1],
        )

        gamma = self.compute_gamma(automation_posterior, F_t, w_t)
        G = C_a - gamma - self.cr

        return G

    def per_step_cost(self, F_t, batched_posterior_h1, deferred_task_indices):
        """
        Compute the per-step cost for a given batch of tasks.
        
        Args:
            F_t (int): Fatigue level at time t
            batched_posterior_h1 (list): List of P(H1|observation) values
            deferred_task_indices (list): Indices of tasks deferred to human
            
        Returns:
            tuple: (auto_cost_per_batch, human_cost_per_batch, deferred_cost)
        """
        total_indices = set(range(self.num_tasks_per_batch))
        auto_indices = list(total_indices - set(deferred_task_indices))
        w_t = len(deferred_task_indices)

        # Automation cost for non-deferred tasks
        auto_cost_per_batch = sum([
            min(
                self.ctn + (self.cfn - self.ctn) * batched_posterior_h1[i],
                self.cfp + (self.ctp - self.cfp) * batched_posterior_h1[i],
            )
            for i in auto_indices
        ])

        # Referral cost
        deferred_cost = len(deferred_task_indices) * self.cr

        # Human performance probabilities
        P_h_fp = self.Phfp(F_t, w_t)
        P_h_tp = self.Phtp(F_t, w_t)

        # Human cost for deferred tasks
        human_cost_per_batch = sum([
            ((1 - batched_posterior_h1[i]) * (P_h_fp * self.cfp + (1 - P_h_fp) * self.ctn) +
             batched_posterior_h1[i] * (P_h_tp * self.ctp + (1 - P_h_tp) * self.cfn))
            for i in deferred_task_indices
        ])

        return auto_cost_per_batch, human_cost_per_batch, deferred_cost

    def compute_cstar(self, F_t, w_t, batched_posterior_h0, batched_posterior_h1):
        """
        Compute the optimal cost c*.
        
        Args:
            F_t (int): Fatigue state at time t
            w_t (float): Taskload at time t
            batched_posterior_h0 (list): List of P(H0|observation) values
            batched_posterior_h1 (list): List of P(H1|observation) values
            
        Returns:
            tuple: (c_star, deferred_indices, sum_G)
        """
        sum_C_a = sum([
            min(
                self.ctn + (self.cfn - self.ctn) * batched_posterior_h1[i],
                self.cfp + (self.ctp - self.cfp) * batched_posterior_h1[i],
            )
            for i in range(self.num_tasks_per_batch)
        ])

        deferred_indices, sum_G = self.algorithm_1_per_wl(
            batched_posterior_h0, batched_posterior_h1, w_t, F_t
        )

        c_star = sum_C_a - sum_G
        return c_star, deferred_indices, sum_G

    def algorithm_1_per_wl(self, batched_posterior_h0, batched_posterior_h1, w_t, F_t):
        """
        Execute Algorithm 1 for a given workload.
        
        Args:
            batched_posterior_h0 (list): List of P(H0|observation) values
            batched_posterior_h1 (list): List of P(H1|observation) values
            w_t (float): Taskload at time t
            F_t (int): Fatigue state at time t
            
        Returns:
            tuple: (human_allocation_indices, G_bar_w)
        """
        # Compute G values for all tasks
        G_vals = [
            self.compute_G([batched_posterior_h0[k], batched_posterior_h1[k]], F_t, w_t)
            for k in range(self.num_tasks_per_batch)
        ]

        # Sort tasks by G value in descending order
        sorted_indices = sorted(range(len(G_vals)), key=lambda i: G_vals[i], reverse=True)

        # Allocate first w_t tasks to human
        wk = int(w_t)
        human_allocation_indices = sorted_indices[:wk]

        # Compute total cost reduction
        G_bar_w = sum([G_vals[k] for k in human_allocation_indices])

        return human_allocation_indices, G_bar_w

    def compute_kesav_policy(self, F_t, batched_posterior_h0, batched_posterior_h1):
        """
        Compute the optimal policy using Kesav's algorithm.
        
        Args:
            F_t (int): Fatigue state at time t
            batched_posterior_h0 (list): List of P(H0|observation) values
            batched_posterior_h1 (list): List of P(H1|observation) values
            
        Returns:
            tuple: (w_t_star, final_human_indices)
        """
        all_gbar_w = []
        all_human_indices = []
        
        for w_t in range(self.num_tasks_per_batch):
            human_allocation_indices, G_bar_w = self.algorithm_1_per_wl(
                batched_posterior_h0, batched_posterior_h1, w_t, F_t
            )
            all_gbar_w.append(G_bar_w)
            all_human_indices.append(human_allocation_indices)

        # Find optimal workload
        max_idx = np.argmax(all_gbar_w)
        w_t_star = max_idx
        final_human_indices = all_human_indices[max_idx]

        return w_t_star, final_human_indices


