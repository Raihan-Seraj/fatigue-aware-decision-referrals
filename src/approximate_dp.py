from utils import Utils
from run_evaluation import Evaluations
from envs.fatigue_model_1 import FatigueMDP
from tqdm import tqdm
import json
import os
import pickle
import argparse
import contextlib



def approximate_dynamic_program(T, num_expectation_samples, ut):
    """
    Compute the approximate dynamic programming algorithm for fatigue-aware task allocation.
    
    This function implements backward induction to compute the optimal value function
    V_t^a(F_t^a) that represents the expected cost-to-go from time t given fatigue state F_t.
    
    Args:
        T (int): The total time horizon
        num_expectation_samples (int): Number of expectation samples to approximate 
                                     the expectation with respect to Y^a_{t+1}
        ut (Utils): Utility object from the Utils class containing problem parameters
        
    Returns:
        list: V_post - List of dictionaries where V_post[t][fatigue_idx] contains
              the optimal expected cost-to-go from time t at fatigue state fatigue_idx
              
    Mathematical Details:
        - Terminal condition: V_T(F_T) = E[min_w C*(F_T, w, Y_T)]
        - Bellman equation: V_t(F_t) = E[min_w {C*(F_t, w, Y_t) + E[V_{t+1}(F_{t+1}) | F_t, w]}]
    """
    # Load the fatigue model and get possible fatigue states
    env = FatigueMDP()
    fatigue_states = env.fatigue_states
   

    # Initialize value function: V_post[t][fatigue_idx] = V_t^a(F_t^a)
    V_post = [{} for _ in range(T + 1)]

    # Compute terminal value function V_T(F_T)
    print("Computing terminal value function...")
    for idx_F_T, F_T in enumerate(fatigue_states):
        expected_terminal_value = _compute_expected_terminal_value(
            F_T, num_expectation_samples, ut
        )
        V_post[T][idx_F_T] = expected_terminal_value

    # Backward induction: compute V_t for t = T-1, T-2, ..., 0
    print("Performing backward iteration...")
    for t in tqdm(range(T - 1, -1, -1), desc="Computing value function"):
        for idx_F_t, F_t in enumerate(fatigue_states):
            expected_value = _compute_expected_value_at_time_t(
                F_t, idx_F_t, t, num_expectation_samples, ut, env, V_post
            )
            V_post[t][idx_F_t] = expected_value

    return V_post


def _compute_expected_terminal_value(F_T, num_expectation_samples, ut):
    """
    Compute the expected terminal value for a given fatigue state.
    
    Args:
        F_T (int): Terminal fatigue state
        num_expectation_samples (int): Number of Monte Carlo samples
        ut (Utils): Utility object
        
    Returns:
        float: Expected terminal value
    """
    expected_terminal_value = 0.0

    for _ in range(num_expectation_samples):
        # Generate automation observations and posterior probabilities
        _, batched_posterior_h0_T, batched_posterior_h1_T = ut.get_auto_obs()

        # Find the minimum cost over all possible workload allocations
        best_action_value = min(
            ut.compute_cstar(F_T, w_t, batched_posterior_h0_T, batched_posterior_h1_T)[0]
            for w_t in range(ut.num_tasks_per_batch)
        )

        expected_terminal_value += best_action_value

    return expected_terminal_value / num_expectation_samples


def _compute_expected_value_at_time_t(F_t, idx_F_t, t, num_expectation_samples, ut, env, V_post):
    """
    Compute the expected value function at time t for a given fatigue state.
    
    Args:
        F_t (int): Current fatigue state
        idx_F_t (int): Index of current fatigue state
        t (int): Current time step
        num_expectation_samples (int): Number of Monte Carlo samples
        ut (Utils): Utility object
        env (FatigueMDP): Fatigue environment
        V_post (list): Value function being computed
        
    Returns:
        float: Expected value at time t
    """
    expected_value = 0.0

    for _ in range(num_expectation_samples):
        # Generate automation observations and posterior probabilities
        _, batched_posterior_h0_t, batched_posterior_h1_t = ut.get_auto_obs()

        # Find the minimum cost over all possible workload allocations
        best_action_value = min(
            _compute_action_value(
                F_t, idx_F_t, w_t, t, batched_posterior_h0_t, batched_posterior_h1_t,
                ut, env, V_post
            )
            for w_t in range(ut.num_tasks_per_batch)
        )

        expected_value += best_action_value

    return expected_value / num_expectation_samples


def _compute_action_value(F_t, idx_F_t, w_t, t, batched_posterior_h0_t, batched_posterior_h1_t, ut, env, V_post):
    """
    Compute the action value for a specific workload allocation.
    
    Args:
        F_t (int): Current fatigue state
        idx_F_t (int): Index of current fatigue state
        w_t (int): Workload allocation (number of tasks to human)
        t (int): Current time step
        batched_posterior_h0_t (list): Posterior probabilities for H0
        batched_posterior_h1_t (list): Posterior probabilities for H1
        ut (Utils): Utility object
        env (FatigueMDP): Fatigue environment
        V_post (list): Value function being computed
        
    Returns:
        float: Action value for the given workload allocation
    """
    # Compute immediate cost
    cstar, _, _ = ut.compute_cstar(F_t, w_t, batched_posterior_h0_t, batched_posterior_h1_t)

    # Compute expected future cost
    w_t_discrete = ut.discretize_taskload(w_t)
    expected_future_cost = sum(
        env.P[w_t_discrete][idx_F_t, idx_F_next] * V_post[t + 1][idx_F_next]
        for idx_F_next in range(len(env.fatigue_states))
    )

    return cstar + expected_future_cost


def run_approximate_dynamic_program(args):
    """
    Execute the approximate dynamic programming algorithm with given parameters.
    
    This function sets up the directory structure, saves parameters, runs the ADP algorithm,
    and saves the computed value function.
    
    Args:
        args (argparse.Namespace): Configuration parameters containing:
            - Fatigue model parameters (alpha_tp, beta_tp, etc.)
            - ADP parameters (num_expectation_samples, horizon)
            - Cost parameters (ctp, ctn, cfp, cfn, cr)
            - Problem setup (num_tasks_per_batch, prior, etc.)
            
    Returns:
        None
        
    Side Effects:
        - Creates directory structure for results
        - Saves parameters as JSON file
        - Saves computed value function as pickle file
    """
    # Create results directory path
    path_name = _create_results_path(args)
    
    # Create directory if it doesn't exist
    if not os.path.exists(path_name):
        with contextlib.suppress(FileExistsError):
            os.makedirs(path_name, exist_ok=True)

    # Save parameters
    _save_parameters(args, path_name)

    # Initialize utility object and run ADP
    ut = Utils(args)
    print(f"Running ADP with T={args.horizon}, samples={args.num_expectation_samples}")
    
    V_func = approximate_dynamic_program(args.horizon, args.num_expectation_samples, ut)

    # Save value function
    with open(path_name + 'V_func.pkl', 'wb') as file:
        pickle.dump(V_func, file)
    
    print(f"Value function saved to {path_name}V_func.pkl")


def _create_results_path(args):
    """
    Create the results directory path based on parameters.
    
    Args:
        args (argparse.Namespace): Configuration parameters
        
    Returns:
        str: Full path to results directory
    """
    return (f"{args.results_path}/")


def _save_parameters(args, path_name):
    """
    Save configuration parameters to JSON file.
    
    Args:
        args (argparse.Namespace): Configuration parameters
        path_name (str): Directory path to save parameters
        
    Returns:
        None
    """
    args_dict = vars(args)
    with open(path_name + 'params.json', 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)


def main():
    """
    Main function to parse arguments and execute the approximate dynamic programming pipeline.
    
    This function:
    1. Parses command line arguments
    2. Runs the ADP algorithm (unless eval_only flag is set)
    3. Runs performance evaluation
    4. Computes and displays performance metrics
    
    Returns:
        None
    """
    parser = _create_argument_parser()
    args = parser.parse_args()

    # Run ADP algorithm unless evaluation-only mode
    if not args.run_eval_only:
        print("Running Approximate Dynamic Programming...")
        run_approximate_dynamic_program(args)

    # Run evaluation
    print("Running evaluation for the computed value function...")
    evaluator = Evaluations(args)
    evaluator.run_perf_eval()

    # Compute and display performance
    print("Computing the performance...")
    evaluator.compute_performance()


def _create_argument_parser():
    """
    Create and configure argument parser for command line interface.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Approximate Dynamic Programming for Fatigue-Aware Task Allocation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ADP Algorithm Parameters
    parser.add_argument('--num_expectation_samples', type=int, default=500,
                       help='Number of expectation samples for approximate DP')
    parser.add_argument('--horizon', type=int, default=20,
                       help='Time horizon length')

    # Problem Setup Parameters
    parser.add_argument('--prior', default=[0.5, 0.5], nargs=2, type=float,
                       help='Prior probabilities [P(H0), P(H1)]')
    parser.add_argument('--num_tasks_per_batch', type=int, default=20,
                       help='Number of tasks in each batch')
    parser.add_argument('--num_bins_fatigue', type=int, default=10,
                       help='Number of bins for fatigue discretization')
    parser.add_argument('--sigma_a', type=float, default=2.3,
                       help='Standard deviation of automation observation noise')

    # Hypothesis Values
    parser.add_argument('--H0', type=int, default=0,
                       help='Value of null hypothesis')
    parser.add_argument('--H1', type=int, default=3,
                       help='Value of alternative hypothesis')

    # Cost Parameters
    parser.add_argument('--ctp', type=float, default=0.0,
                       help='Cost of true positive')
    parser.add_argument('--ctn', type=float, default=0.0,
                       help='Cost of true negative')
    parser.add_argument('--cfp', type=float, default=1.0,
                       help='Cost of false positive')
    parser.add_argument('--cfn', type=float, default=1.0,
                       help='Cost of false negative')
    parser.add_argument('--cr', type=float, default=0.0,
                       help='Cost of referral to human')

    # Fatigue Model Parameters
    parser.add_argument('--alpha_tp', type=float, default=0.087,
                       help='Alpha parameter for true positive probability')
    parser.add_argument('--beta_tp', type=float, default=0.043,
                       help='Beta parameter for true positive probability')
    parser.add_argument('--alpha_fp', type=float, default=0.1,
                       help='Alpha parameter for false positive probability')
    parser.add_argument('--beta_fp', type=float, default=0.033,
                       help='Beta parameter for false positive probability')
    parser.add_argument('--gamma_tp', type=float, default=0.0,
                       help='Gamma parameter for true positive probability')
    parser.add_argument('--gamma_fp', type=float, default=0.0,
                       help='Gamma parameter for false positive probability')

    # Execution Parameters
    parser.add_argument('--results_path', type=str, default='results/',
                       help='Directory to save results')
    parser.add_argument('--run_eval_only', type=bool, default=False,
                       help='Run evaluation only, skip ADP computation')
    parser.add_argument('--num_eval_runs', type=int, default=1000,
                       help='Number of Monte Carlo runs for evaluation')

    return parser


if __name__ == "__main__":
    main()




