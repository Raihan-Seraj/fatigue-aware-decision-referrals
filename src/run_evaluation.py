import os
import pickle
import contextlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import Utils
from envs.fatigue_model_1 import FatigueMDP


# Set matplotlib to non-interactive backend for server environments
import matplotlib
matplotlib.use('Agg')


class Evaluations:
    """
    Evaluation class for comparing ADP and myopic algorithms performance.
    
    This class provides comprehensive evaluation methods for fatigue-aware
    decision referral algorithms, including performance metrics, plotting,
    and robustness analysis.
    """

    def __init__(self, args):
        """
        Initialize the Evaluations class.
        
        Args:
            args (argparse.Namespace): Configuration parameters containing:
                - results_path: Path to save results
                - num_tasks_per_batch: Number of tasks per batch
                - alpha_tp, beta_tp, alpha_fp, beta_fp: Fatigue model parameters

                - horizon: Time horizon for evaluation
                - num_eval_runs: Number of Monte Carlo evaluation runs
        """
        self.args = args
        
        
        # Initialize appropriate fatigue model
        
        self.env = FatigueMDP()
            
        self.fatigue_states = self.env.fatigue_states
        self.global_path = self._create_global_path()
        self.num_tasks_per_batch = args.num_tasks_per_batch

    def _create_global_path(self):
        """
        Create the global path for saving results based on model parameters.
        
        Returns:
            str: Full path to results directory
        """
        base_path = f"{self.args.results_path}/"
        
        return base_path 

    def compute_adp_solution(self, batched_posterior_h0, batched_posterior_h1, F_t, idx_F_t, V_bar, ut):
        """
        Compute the optimal workload using the ADP solution.
        
        This method finds the optimal workload allocation by minimizing the sum of
        immediate cost and expected future cost based on the value function.
        
        Args:
            batched_posterior_h0 (list): Posterior probabilities for H0
            batched_posterior_h1 (list): Posterior probabilities for H1
            F_t (int): Current fatigue state
            idx_F_t (int): Index of current fatigue state
            V_bar (dict): Value function at current time
            ut (Utils): Utility object
            
        Returns:
            tuple: (optimal_workload, deferred_task_indices)
        """
        all_costs = []
        all_deferred_indices = []

        for w_t in range(self.num_tasks_per_batch):
            # Discretize workload based on fatigue model
            w_t_discrete = self._get_discrete_workload(w_t, ut)
            
            # Compute immediate cost
            cstar, deferred_indices, _ = ut.compute_cstar(
                F_t, w_t, batched_posterior_h0, batched_posterior_h1
            )
            
            # Compute expected future cost
            expected_future_cost = sum(
                self.env.P[w_t_discrete][idx_F_t, idx_F_next] * V_bar[idx_F_next]
                for idx_F_next in range(len(self.fatigue_states))
            )
            
            total_cost = cstar + expected_future_cost
            all_costs.append(total_cost)
            all_deferred_indices.append(deferred_indices)

        # Find optimal workload
        min_idx = np.argmin(all_costs)
        return min_idx, all_deferred_indices[min_idx]

    def _get_discrete_workload(self, w_t, ut):
        """
        Get discrete workload based on fatigue model type.
        
        Args:
            w_t (int): Workload value
            ut (Utils): Utility object
            
        Returns:
            int: Discretized workload
        """
        
        return ut.discretize_taskload(w_t)
        
    def compute_performance(self):
        """
        Compute comprehensive performance comparison between ADP and myopic algorithms.
        
        This method runs Monte Carlo simulations to evaluate both algorithms across
        multiple scenarios and saves detailed cost breakdowns and cumulative costs.
        
        Returns:
            None
            
        Side Effects:
            - Saves cost comparison data to files
            - Saves workload allocation data
            - Saves batch observation data
        """
        print("Computing performance...")
        
        ut = Utils(self.args)
        V_bar = self._load_value_function()
        
        num_runs = self.args.num_eval_runs
        
        # Initialize cost tracking arrays
        cost_data = self._initialize_cost_arrays(num_runs)
        
        # Initialize tracking dictionaries
        all_human_wl_adp = {}
        all_human_wl_k = {}
        all_mega_batch = {}
        run_cum_cost_k = []
        run_cum_cost_adp = []
        
        # Run Monte Carlo simulations
        for run in tqdm(range(num_runs), desc="Running performance evaluation"):
            run_data = self._run_single_performance_evaluation(
                run, ut, V_bar, cost_data
            )
            
            # Store run-specific data
            all_human_wl_adp[f'Run-{run+1}'] = run_data['hum_wl_adp']
            all_human_wl_k[f'Run-{run+1}'] = run_data['hum_wl_k']
            all_mega_batch[f'Run-{run+1}'] = run_data['mega_batch']
            run_cum_cost_k.append(run_data['cum_cost_k'])
            run_cum_cost_adp.append(run_data['cum_cost_adp'])

        # Save all results
        self._save_performance_results(
            cost_data, all_human_wl_adp, all_human_wl_k, 
            all_mega_batch, run_cum_cost_k, run_cum_cost_adp
        )

    def _initialize_cost_arrays(self, num_runs):
        """
        Initialize arrays for tracking costs across runs.
        
        Args:
            num_runs (int): Number of evaluation runs
            
        Returns:
            dict: Dictionary containing initialized cost arrays
        """
        return {
            'auto_cost_k': np.zeros(num_runs),
            'human_cost_k': np.zeros(num_runs),
            'deferred_cost_k': np.zeros(num_runs),
            'auto_cost_adp': np.zeros(num_runs),
            'human_cost_adp': np.zeros(num_runs),
            'deferred_cost_adp': np.zeros(num_runs),
            'auto_cost_adp_new': np.zeros(num_runs),
            'human_cost_adp_new': np.zeros(num_runs),
            'deferred_cost_adp_new': np.zeros(num_runs)
        }

    def _run_single_performance_evaluation(self, run, ut, V_bar, cost_data):
        """
        Run a single performance evaluation simulation.
        
        Args:
            run (int): Current run number
            ut (Utils): Utility object
            V_bar (list): Value function
            cost_data (dict): Cost tracking arrays
            
        Returns:
            dict: Run-specific data including costs and workloads
        """
        # Initialize fatigue states
        F_k = F_adp = 0
        idx_F_k = idx_F_adp = 0
        
        # Initialize cost accumulators
        costs = {
            'auto_k': 0, 'human_k': 0, 'deferred_k': 0,
            'auto_adp': 0, 'human_adp': 0, 'deferred_adp': 0
        }
        
        # Generate observations for entire horizon
        mega_batch = [ut.get_auto_obs() for _ in range(self.args.horizon)]
        
        # Initialize tracking arrays
        hum_wl_adp = np.zeros(self.args.horizon)
        hum_wl_k = np.zeros(self.args.horizon)
        cum_cost_k = []
        cum_cost_adp = []
        
        # Simulate over time horizon
        for t in range(self.args.horizon):
            _, batched_posterior_h0, batched_posterior_h1 = mega_batch[t]
            
            # Compute optimal policies
            wl_k, deferred_idx_k = ut.compute_myopic_policy(
                F_k, batched_posterior_h0, batched_posterior_h1
            )
            wl_adp, deferred_idx_adp = self.compute_adp_solution(
                batched_posterior_h0, batched_posterior_h1, F_adp, idx_F_adp, V_bar[t], ut
            )
            
            # Store workloads
            hum_wl_adp[t] = wl_adp
            hum_wl_k[t] = wl_k
            
            # Compute step costs
            step_costs = self._compute_step_costs(
                ut, F_k, F_adp, batched_posterior_h1, deferred_idx_k, deferred_idx_adp
            )
            
            # Accumulate costs
            for key, value in step_costs.items():
                costs[key] += value
            
            # Track cumulative costs
            cum_cost_k.append(sum(step_costs[k] for k in ['auto_k', 'human_k', 'deferred_k']))
            cum_cost_adp.append(sum(step_costs[k] for k in ['auto_adp', 'human_adp', 'deferred_adp']))
            
            # Update fatigue states
            F_k, idx_F_k = self._update_fatigue_state(F_k, wl_k, ut)
            F_adp, idx_F_adp = self._update_fatigue_state(F_adp, wl_adp, ut)
        
        # Store final costs
        self._store_run_costs(run, costs, cost_data)
        
        return {
            'hum_wl_adp': hum_wl_adp,
            'hum_wl_k': hum_wl_k,
            'mega_batch': mega_batch,
            'cum_cost_k': cum_cost_k,
            'cum_cost_adp': cum_cost_adp
        }

    def _compute_step_costs(self, ut, F_k, F_adp, batched_posterior_h1, deferred_idx_k, deferred_idx_adp):
        """
        Compute costs for a single time step.
        
        Args:
            ut (Utils): Utility object
            F_k (int): myopic algorithm fatigue state
            F_adp (int): ADP algorithm fatigue state
            batched_posterior_h1 (list): Posterior probabilities for H1
            deferred_idx_k (list): myopic deferred task indices
            deferred_idx_adp (list): ADP deferred task indices
            
        Returns:
            dict: Step costs for both algorithms
        """
        a_cost_adp, h_cost_adp, def_cost_adp = ut.per_step_cost(
            F_adp, batched_posterior_h1, deferred_idx_adp
        )
        a_cost_k, h_cost_k, def_cost_k = ut.per_step_cost(
            F_k, batched_posterior_h1, deferred_idx_k
        )
        
        return {
            'auto_adp': a_cost_adp, 'human_adp': h_cost_adp, 'deferred_adp': def_cost_adp,
            'auto_k': a_cost_k, 'human_k': h_cost_k, 'deferred_k': def_cost_k
        }

    def _update_fatigue_state(self, F_current, workload, ut):
        """
        Update fatigue state based on workload.
        
        Args:
            F_current (int): Current fatigue state
            workload (int): Assigned workload
            ut (Utils): Utility object
            
        Returns:
            tuple: (new_fatigue_state, new_fatigue_index)
        """
        workload_discrete = self._get_discrete_workload(workload, ut)
        return self.env.next_state(F_current, workload_discrete)

    def _store_run_costs(self, run, costs, cost_data):
        """
        Store costs from a single run into the cost tracking arrays.
        
        Args:
            run (int): Run number
            costs (dict): Costs from the run
            cost_data (dict): Cost tracking arrays
        """
        cost_data['auto_cost_adp'][run] = costs['auto_adp']
        cost_data['human_cost_adp'][run] = costs['human_adp']
        cost_data['deferred_cost_adp'][run] = costs['deferred_adp']
        cost_data['auto_cost_k'][run] = costs['auto_k']
        cost_data['human_cost_k'][run] = costs['human_k']
        cost_data['deferred_cost_k'][run] = costs['deferred_k']

    def _load_value_function(self):
        """
        Load the value function from pickle file.
        
        Returns:
            list: Value function
        """
        with open(self.global_path + 'V_func.pkl', 'rb') as file:
            return pickle.load(file)

    def _save_performance_results(self, cost_data, all_human_wl_adp, all_human_wl_k, 
                                all_mega_batch, run_cum_cost_k, run_cum_cost_adp):
        """
        Save all performance evaluation results to files.
        
        Args:
            cost_data (dict): Cost arrays
            all_human_wl_adp (dict): ADP workload data
            all_human_wl_k (dict): myopic workload data
            all_mega_batch (dict): Observation batch data
            run_cum_cost_k (list): Cumulative costs for myopic
            run_cum_cost_adp (list): Cumulative costs for ADP
        """
        path1 = self.global_path + 'cost_comparison/'
        if not os.path.exists(path1):
            with contextlib.suppress(FileExistsError):
                os.makedirs(path1, exist_ok=True)
        
        # Save pickle files
        pickle_files = {
            'all_cum_cost_k.pkl': run_cum_cost_k,
            'all_cum_cost_adp.pkl': run_cum_cost_adp,
            'all_human_wl_adp.pkl': all_human_wl_adp,
            'all_human_wl_k.pkl': all_human_wl_k,
            'all_mega_batch.pkl': all_mega_batch
        }
        
        for filename, data in pickle_files.items():
            with open(path1 + filename, 'wb') as file:
                pickle.dump(data, file)
        
        # Save numpy arrays
        numpy_files = {
            'all_auto_cost_adp.npy': cost_data['auto_cost_adp'],
            'all_human_cost_adp.npy': cost_data['human_cost_adp'],
            'all_deferred_cost_adp.npy': cost_data['deferred_cost_adp'],
            'all_auto_cost_adp_new.npy': cost_data['auto_cost_adp_new'],
            'all_human_cost_adp_new.npy': cost_data['human_cost_adp_new'],
            'all_deferred_cost_adp_new.npy': cost_data['deferred_cost_adp_new'],
            'all_auto_cost_k.npy': cost_data['auto_cost_k'],
            'all_human_cost_k.npy': cost_data['human_cost_k'],
            'all_deferred_cost_k.npy': cost_data['deferred_cost_k']
        }
        
        for filename, data in numpy_files.items():
            np.save(os.path.join(path1, filename), data)

    def run_evaluation(self):
        """
        Run a single evaluation trajectory comparing ADP and myopic algorithms.
        
        Returns:
            tuple: (fatigue_evolution_myopic, fatigue_evolution_adp, 
                   taskload_evolution_myopic, taskload_evolution_adp)
        """
        ut = Utils(self.args)
        V_bar = self._load_value_function()
        
        # Initialize states
        F_k = F_adp = 0
        idx_F_k = idx_F_adp = 0
        
        # Initialize tracking lists
        fatigue_evolution_myopic = []
        fatigue_evolution_adp = []
        taskload_evolution_myopic = []
        taskload_evolution_adp = []
        
        # Generate observations for entire horizon
        mega_obs = [ut.get_auto_obs() for _ in range(self.args.horizon)]
        
        for t in range(self.args.horizon):
            # Record current fatigue states
            fatigue_evolution_myopic.append(F_k)
            fatigue_evolution_adp.append(F_adp)
            
            # Get observations
            _, batched_posterior_h0, batched_posterior_h1 = mega_obs[t]
            
            # Compute optimal policies
            wl_k, _ = ut.compute_myopic_policy(F_k, batched_posterior_h0, batched_posterior_h1)
            wl_adp, _ = self.compute_adp_solution(
                batched_posterior_h0, batched_posterior_h1, F_adp, idx_F_adp, V_bar[t], ut
            )
            
            # Record workloads
            taskload_evolution_myopic.append(wl_k)
            taskload_evolution_adp.append(wl_adp)
            
            # Update fatigue states
            F_k, idx_F_k = self._update_fatigue_state(F_k, wl_k, ut)
            F_adp, idx_F_adp = self._update_fatigue_state(F_adp, wl_adp, ut)
        
        return (fatigue_evolution_myopic, fatigue_evolution_adp, 
                taskload_evolution_myopic, taskload_evolution_adp)

    def run_perf_eval(self):
        """
        Run performance evaluation and generate comparison plots.
        
        This method runs multiple evaluation trajectories, computes statistics,
        and generates plots comparing fatigue and workload evolution.
        
        Returns:
            None
            
        Side Effects:
            - Saves trajectory data to files
            - Generates and saves comparison plots
        """
        num_runs = self.args.num_eval_runs
        
        # Initialize tracking lists
        all_fatigue_myopic = []
        all_fatigue_adp = []
        all_taskload_myopic = []
        all_taskload_adp = []
        
        # Run evaluations
        for run in tqdm(range(num_runs), desc="Running trajectory evaluations"):
            trajectories = self.run_evaluation()
            all_fatigue_myopic.append(trajectories[0])
            all_fatigue_adp.append(trajectories[1])
            all_taskload_myopic.append(trajectories[2])
            all_taskload_adp.append(trajectories[3])
        
        # Save trajectory data
        self._save_trajectory_data(
            all_fatigue_myopic, all_fatigue_adp, 
            all_taskload_myopic, all_taskload_adp
        )
        
        # Generate plots
        self._generate_comparison_plots(
            all_fatigue_myopic, all_fatigue_adp,
            all_taskload_myopic, all_taskload_adp
        )

    def _save_trajectory_data(self, all_fatigue_myopic, all_fatigue_adp, 
                            all_taskload_myopic, all_taskload_adp):
        """
        Save trajectory data to pickle files.
        
        Args:
            all_fatigue_myopic (list): Fatigue trajectories for myopic algorithm
            all_fatigue_adp (list): Fatigue trajectories for ADP algorithm
            all_taskload_myopic (list): Taskload trajectories for myopic algorithm
            all_taskload_adp (list): Taskload trajectories for ADP algorithm
        """
        path_name = self.global_path + '/plot_analysis/'
        if not os.path.exists(path_name):
            with contextlib.suppress(FileExistsError):
                os.makedirs(path_name, exist_ok=True)
        
        data_files = {
            'all_fatigue_k.pkl': all_fatigue_myopic,
            'all_fatigue_adp.pkl': all_fatigue_adp,
            'all_taskload_k.pkl': all_taskload_myopic,
            'all_taskload_adp.pkl': all_taskload_adp
        }
        
        for filename, data in data_files.items():
            with open(path_name + filename, 'wb') as file:
                pickle.dump(data, file)

    def _generate_comparison_plots(self, all_fatigue_myopic, all_fatigue_adp,
                                 all_taskload_myopic, all_taskload_adp):
        """
        Generate comparison plots for fatigue and taskload evolution.
        
        Args:
            all_fatigue_myopic (list): Fatigue trajectories for myopic algorithm
            all_fatigue_adp (list): Fatigue trajectories for ADP algorithm
            all_taskload_myopic (list): Taskload trajectories for myopic algorithm
            all_taskload_adp (list): Taskload trajectories for ADP algorithm
        """
        path_name = self.global_path + '/plot_analysis/'
        time_steps = np.arange(1, self.args.horizon + 1, 1)
        
        # Compute statistics for fatigue
        fatigue_stats = self._compute_trajectory_statistics(all_fatigue_myopic, all_fatigue_adp)
        
        # Plot fatigue evolution
        self._plot_evolution_comparison(
            time_steps, fatigue_stats, 'Fatigue Level', 
            path_name + 'fatigue_evolution_comparison.pdf'
        )
        
        # Compute statistics for taskload
        taskload_stats = self._compute_trajectory_statistics(all_taskload_myopic, all_taskload_adp)
        
        # Plot taskload evolution
        self._plot_evolution_comparison(
            time_steps, taskload_stats, 'Workload level',
            path_name + 'taskload_evolution_comparison.pdf'
        )

    def _compute_trajectory_statistics(self, myopic_data, adp_data):
        """
        Compute median and percentile statistics for trajectory data.
        
        Args:
            myopic_data (list): Trajectory data for myopic algorithm
            adp_data (list): Trajectory data for ADP algorithm
            
        Returns:
            dict: Statistics including medians and percentiles
        """
        return {
            'median_k': np.median(myopic_data, axis=0),
            'lower_k': np.percentile(myopic_data, 25, axis=0),
            'upper_k': np.percentile(myopic_data, 75, axis=0),
            'median_adp': np.median(adp_data, axis=0),
            'lower_adp': np.percentile(adp_data, 25, axis=0),
            'upper_adp': np.percentile(adp_data, 75, axis=0)
        }

    def _plot_evolution_comparison(self, time_steps, stats, ylabel, filename):
        """
        Create and save evolution comparison plot.
        
        Args:
            time_steps (np.array): Time step array
            stats (dict): Statistics dictionary
            ylabel (str): Y-axis label
            filename (str): Output filename
        """
        plt.figure(figsize=(10, 6))
        
        # Plot myopic algorithm
        plt.step(time_steps, stats['median_k'], color='black', where='post', 
                label='K-Algorithm', linewidth=2)
        plt.fill_between(time_steps, stats['lower_k'], stats['upper_k'],
                        step='post', alpha=0.2, color='black')
        
        # Plot ADP algorithm
        plt.step(time_steps, stats['median_adp'], color='orange', where='post', 
                label='ADP', linewidth=2)
        plt.fill_between(time_steps, stats['lower_adp'], stats['upper_adp'],
                        step='post', alpha=0.2, color='orange')
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Time')
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def eval_single_run(self, initial_fatigue_state, initial_fatigue_index):
        """
        Evaluate a single run with specified initial fatigue state.
        
        Args:
            initial_fatigue_state (int): Initial fatigue state value
            initial_fatigue_index (int): Initial fatigue state index
            
        Returns:
            tuple: (fatigue_evolution_myopic, fatigue_evolution_adp, 
                   taskload_evolution_myopic, taskload_evolution_adp)
        """
        ut = Utils(self.args)
        V_bar = self._load_value_function()
        
        # Initialize states with specified values
        F_k = F_adp = initial_fatigue_state
        idx_F_k = idx_F_adp = initial_fatigue_index
        
        # Initialize tracking lists
        fatigue_evolution_myopic = []
        fatigue_evolution_adp = []
        taskload_evolution_myopic = []
        taskload_evolution_adp = []
        
        # Generate observations for entire horizon
        mega_obs = [ut.get_auto_obs() for _ in range(self.args.horizon)]
        
        for t in range(self.args.horizon):
            # Record current fatigue states
            fatigue_evolution_myopic.append(F_k)
            fatigue_evolution_adp.append(F_adp)
            
            # Get observations
            _, batched_posterior_h0, batched_posterior_h1 = mega_obs[t]
            
            # Compute optimal policies
            wl_k, _ = ut.compute_myopic_policy(F_k, batched_posterior_h0, batched_posterior_h1)
            wl_adp, _ = self.compute_adp_solution(
                batched_posterior_h0, batched_posterior_h1, F_adp, idx_F_adp, V_bar[t], ut
            )
            
            # Record workloads
            taskload_evolution_myopic.append(wl_k)
            taskload_evolution_adp.append(wl_adp)
            
            # Update fatigue states
            F_k, idx_F_k = self._update_fatigue_state(F_k, wl_k, ut)
            F_adp, idx_F_adp = self._update_fatigue_state(F_adp, wl_adp, ut)
        
        return (fatigue_evolution_myopic, fatigue_evolution_adp, 
                taskload_evolution_myopic, taskload_evolution_adp)

    def eval_single_run_perturbed_fatigue(self, original_fatigue_mdp, perturbed_fatigue_mdp):
        """
        Evaluate a single run with perturbed fatigue model for robustness analysis.
        
        Args:
            original_fatigue_mdp: Original fatigue MDP (unused but kept for compatibility)
            perturbed_fatigue_mdp: Perturbed fatigue MDP for simulation
            
        Returns:
            tuple: (fatigue_evolution_myopic, fatigue_evolution_adp, 
                   taskload_evolution_myopic, taskload_evolution_adp)
        """
        ut = Utils(self.args)
        V_bar = self._load_value_function()
        
        # Initialize states
        F_k = F_adp = 0
        idx_F_k = idx_F_adp = 0
        
        # Initialize tracking lists
        fatigue_evolution_myopic = []
        fatigue_evolution_adp = []
        taskload_evolution_myopic = []
        taskload_evolution_adp = []
        
        # Generate observations for entire horizon
        mega_obs = [ut.get_auto_obs() for _ in range(self.args.horizon)]
        
        for t in range(self.args.horizon):
            # Record current fatigue states
            fatigue_evolution_myopic.append(F_k)
            fatigue_evolution_adp.append(F_adp)
            
            # Get observations
            _, batched_posterior_h0, batched_posterior_h1 = mega_obs[t]
            
            # Compute optimal policies
            wl_k, _ = ut.compute_myopic_policy(F_k, batched_posterior_h0, batched_posterior_h1)
            wl_adp, _ = self.compute_adp_solution(
                batched_posterior_h0, batched_posterior_h1, F_adp, idx_F_adp, V_bar[t], ut
            )
            
            # Record workloads
            taskload_evolution_myopic.append(wl_k)
            taskload_evolution_adp.append(wl_adp)
            
            # Update fatigue states using perturbed model
            wl_k_discrete = ut.discretize_taskload(wl_k)
            wl_adp_discrete = ut.discretize_taskload(wl_adp)
            F_k, idx_F_k = perturbed_fatigue_mdp.next_state(F_k, wl_k_discrete)
            F_adp, idx_F_adp = perturbed_fatigue_mdp.next_state(F_adp, wl_adp_discrete)
        
        return (fatigue_evolution_myopic, fatigue_evolution_adp, 
                taskload_evolution_myopic, taskload_evolution_adp)

    def compute_performance_perturbed_fatigue(self, perturbed_fatigue_mdp, path_to_save):
        """
        Compute performance with perturbed fatigue model for robustness analysis.
        
        Args:
            perturbed_fatigue_mdp: Perturbed fatigue MDP
            path_to_save (str): Path to save results
            
        Returns:
            None
            
        Side Effects:
            - Saves robustness analysis results
            - Saves perturbed transition matrices
        """
        print("Computing performance with perturbed fatigue model...")
        
        ut = Utils(self.args)
        V_bar = self._load_value_function()
        
        num_runs = self.args.num_eval_runs
        cost_data = self._initialize_cost_arrays(num_runs)
        
        # Initialize tracking dictionaries
        all_human_wl_adp = {}
        all_human_wl_k = {}
        all_mega_batch = {}
        run_cum_cost_k = []
        run_cum_cost_adp = []
        
        # Run Monte Carlo simulations with perturbed model
        for run in tqdm(range(num_runs), desc="Running perturbed performance evaluation"):
            run_data = self._run_single_performance_evaluation_perturbed(
                run, ut, V_bar, cost_data, perturbed_fatigue_mdp
            )
            
            # Store run-specific data
            all_human_wl_adp[f'Run-{run+1}'] = run_data['hum_wl_adp']
            all_human_wl_k[f'Run-{run+1}'] = run_data['hum_wl_k']
            all_mega_batch[f'Run-{run+1}'] = run_data['mega_batch']
            run_cum_cost_k.append(run_data['cum_cost_k'])
            run_cum_cost_adp.append(run_data['cum_cost_adp'])
        
        # Save results with perturbed model data
        self._save_perturbed_performance_results(
            path_to_save, perturbed_fatigue_mdp, cost_data, 
            all_human_wl_adp, all_human_wl_k, all_mega_batch,
            run_cum_cost_k, run_cum_cost_adp
        )

    def _run_single_performance_evaluation_perturbed(self, run, ut, V_bar, cost_data, perturbed_mdp):
        """
        Run a single performance evaluation with perturbed fatigue model.
        
        Args:
            run (int): Current run number
            ut (Utils): Utility object
            V_bar (list): Value function
            cost_data (dict): Cost tracking arrays
            perturbed_mdp: Perturbed fatigue MDP
            
        Returns:
            dict: Run-specific data including costs and workloads
        """
        # Initialize fatigue states
        F_k = F_adp = 0
        idx_F_k = idx_F_adp = 0
        
        # Initialize cost accumulators
        costs = {
            'auto_k': 0, 'human_k': 0, 'deferred_k': 0,
            'auto_adp': 0, 'human_adp': 0, 'deferred_adp': 0
        }
        
        # Generate observations for entire horizon
        mega_batch = [ut.get_auto_obs() for _ in range(self.args.horizon)]
        
        # Initialize tracking arrays
        hum_wl_adp = np.zeros(self.args.horizon)
        hum_wl_k = np.zeros(self.args.horizon)
        cum_cost_k = []
        cum_cost_adp = []
        
        # Simulate over time horizon
        for t in range(self.args.horizon):
            _, batched_posterior_h0, batched_posterior_h1 = mega_batch[t]
            
            # Compute optimal policies
            wl_k, deferred_idx_k = ut.compute_myopic_policy(
                F_k, batched_posterior_h0, batched_posterior_h1
            )
            wl_adp, deferred_idx_adp = self.compute_adp_solution(
                batched_posterior_h0, batched_posterior_h1, F_adp, idx_F_adp, V_bar[t], ut
            )
            
            # Store workloads
            hum_wl_adp[t] = wl_adp
            hum_wl_k[t] = wl_k
            
            # Compute step costs
            step_costs = self._compute_step_costs(
                ut, F_k, F_adp, batched_posterior_h1, deferred_idx_k, deferred_idx_adp
            )
            
            # Accumulate costs
            for key, value in step_costs.items():
                costs[key] += value
            
            # Track cumulative costs
            cum_cost_k.append(sum(step_costs[k] for k in ['auto_k', 'human_k', 'deferred_k']))
            cum_cost_adp.append(sum(step_costs[k] for k in ['auto_adp', 'human_adp', 'deferred_adp']))
            
            # Update fatigue states using perturbed model
            wl_k_discrete = ut.discretize_taskload(wl_k)
            wl_adp_discrete = ut.discretize_taskload(wl_adp)
            F_k, idx_F_k = perturbed_mdp.next_state(F_k, wl_k_discrete)
            F_adp, idx_F_adp = perturbed_mdp.next_state(F_adp, wl_adp_discrete)
        
        # Store final costs
        self._store_run_costs(run, costs, cost_data)
        
        return {
            'hum_wl_adp': hum_wl_adp,
            'hum_wl_k': hum_wl_k,
            'mega_batch': mega_batch,
            'cum_cost_k': cum_cost_k,
            'cum_cost_adp': cum_cost_adp
        }

    def _save_perturbed_performance_results(self, path_to_save, perturbed_mdp, cost_data,
                                          all_human_wl_adp, all_human_wl_k, all_mega_batch,
                                          run_cum_cost_k, run_cum_cost_adp):
        """
        Save performance results for perturbed fatigue model analysis.
        
        Args:
            path_to_save (str): Base path to save results
            perturbed_mdp: Perturbed fatigue MDP
            cost_data (dict): Cost arrays
            all_human_wl_adp (dict): ADP workload data
            all_human_wl_k (dict): myopic workload data
            all_mega_batch (dict): Observation batch data
            run_cum_cost_k (list): Cumulative costs for myopic
            run_cum_cost_adp (list): Cumulative costs for ADP
        """
        path1 = os.path.join(path_to_save, 'cost_comparison/')
        if not os.path.exists(path1):
            with contextlib.suppress(FileExistsError):
                os.makedirs(path1, exist_ok=True)
        
        # Save perturbed transition matrix
        with open(os.path.join(path1, 'perturbed_transition.pkl'), 'wb') as file:
            pickle.dump(perturbed_mdp.P, file)
        
        # Save pickle files
        pickle_files = {
            'all_cum_cost_k.pkl': run_cum_cost_k,
            'all_cum_cost_adp.pkl': run_cum_cost_adp,
            'all_human_wl_adp.pkl': all_human_wl_adp,
            'all_human_wl_k.pkl': all_human_wl_k,
            'all_mega_batch.pkl': all_mega_batch
        }
        
        for filename, data in pickle_files.items():
            with open(os.path.join(path1, filename), 'wb') as file:
                pickle.dump(data, file)
        
        # Save numpy arrays
        numpy_files = {
            'all_auto_cost_adp.npy': cost_data['auto_cost_adp'],
            'all_human_cost_adp.npy': cost_data['human_cost_adp'],
            'all_deferred_cost_adp.npy': cost_data['deferred_cost_adp'],
            'all_auto_cost_adp_new.npy': cost_data['auto_cost_adp_new'],
            'all_human_cost_adp_new.npy': cost_data['human_cost_adp_new'],
            'all_deferred_cost_adp_new.npy': cost_data['deferred_cost_adp_new'],
            'all_auto_cost_k.npy': cost_data['auto_cost_k'],
            'all_human_cost_k.npy': cost_data['human_cost_k'],
            'all_deferred_cost_k.npy': cost_data['deferred_cost_k']
        }
        
        for filename, data in numpy_files.items():
            np.save(os.path.join(path1, filename), data)


