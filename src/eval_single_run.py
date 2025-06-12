from run_evaluation import Evaluations
from tqdm import tqdm
import os
import pickle
import argparse


def create_argument_parser():
    """
    Create and configure argument parser for single run evaluation.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser with all required parameters
    """
    parser = argparse.ArgumentParser(
        description="Single run evaluation of ADP and Kesav algorithms with different initial fatigue states",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Algorithm Parameters
    parser.add_argument('--num_expectation_samples', type=int, default=500,
                       help='Number of expectation samples for approximate Dynamic Program')
    parser.add_argument('--horizon', type=int, default=10,
                       help='Time horizon length for evaluation')
    parser.add_argument('--num_eval_runs', type=int, default=1000,
                       help='Number of independent Monte Carlo runs for performance evaluation')

    # Problem Setup Parameters
    parser.add_argument('--prior', default=[0.5, 0.5], nargs=2, type=float,
                       help='Prior probabilities [P(H0), P(H1)]')
    parser.add_argument('--num_tasks_per_batch', type=int, default=20,
                       help='Total number of tasks in each batch')
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
                       help='Cost of true positive classification')
    parser.add_argument('--ctn', type=float, default=0.0,
                       help='Cost of true negative classification')
    parser.add_argument('--cfp', type=float, default=1.0,
                       help='Cost of false positive classification')
    parser.add_argument('--cfn', type=float, default=1.0,
                       help='Cost of false negative classification')
    parser.add_argument('--cr', type=float, default=0.0,
                       help='Cost of task deferral to human operator')

    # Fatigue Model Parameters
    parser.add_argument('--alpha_tp', type=float, default=0.087,
                       help='Alpha parameter for true positive probability (fatigue impact)')
    parser.add_argument('--beta_tp', type=float, default=0.043,
                       help='Beta parameter for true positive probability (workload impact)')
    parser.add_argument('--alpha_fp', type=float, default=0.1,
                       help='Alpha parameter for false positive probability (fatigue impact)')
    parser.add_argument('--beta_fp', type=float, default=0.033,
                       help='Beta parameter for false positive probability (workload impact)')

    # Output Parameters
    parser.add_argument('--results_path', type=str, default='results/',
                       help='Base directory for algorithm results')
    parser.add_argument('--run_eval_only', type=bool, default=False,
                       help='Run evaluation only, skip computation')

    return parser


def get_fatigue_state_mapping():
    """
    Get mapping from fatigue state names to values and indices.
    
    Returns:
        dict: Mapping of fatigue state names to (state_value, state_index) tuples
    """
    return {
        'fatigue_high': (2, 2),
        'fatigue_low': (0, 0)
    }


def create_output_directory(base_path):
    """
    Create output directory for single run evaluation results.
    
    Args:
        base_path (str): Base path for output directory
        
    Returns:
        str: Created directory path
        
    Side Effects:
        Creates directory if it doesn't exist
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
        print(f"Created output directory: {base_path}")
    
    return base_path


def save_trajectory_data(output_path, fatigue_kesav, fatigue_adp, taskload_kesav, taskload_adp):
    """
    Save trajectory data to pickle files.
    
    Args:
        output_path (str): Directory path to save files
        fatigue_kesav (list): Fatigue evolution for Kesav algorithm
        fatigue_adp (list): Fatigue evolution for ADP algorithm
        taskload_kesav (list): Taskload evolution for Kesav algorithm
        taskload_adp (list): Taskload evolution for ADP algorithm
        
    Returns:
        None
        
    Side Effects:
        Creates four pickle files with trajectory data
    """
    trajectory_files = {
        'fatigue_evolve_k.pkl': fatigue_kesav,
        'fatigue_evolve_adp.pkl': fatigue_adp,
        'taskload_evolve_k.pkl': taskload_kesav,
        'taskload_evolve_adp.pkl': taskload_adp
    }
    
    for filename, data in trajectory_files.items():
        filepath = os.path.join(output_path, filename)
        with open(filepath, 'wb') as file:
            pickle.dump(data, file)


def run_single_evaluation(evaluator, run_number, fatigue_state_name, base_save_path, fatigue_mapping):
    """
    Run a single evaluation for specified run number and fatigue state.
    
    Args:
        evaluator (Evaluations): Evaluator instance
        run_number (int): Current run number
        fatigue_state_name (str): Name of initial fatigue state
        base_save_path (str): Base path for saving results
        fatigue_mapping (dict): Mapping of fatigue state names to values
        
    Returns:
        None
        
    Side Effects:
        Creates directory and saves trajectory data
    """
    # Create run-specific directory
    run_path = os.path.join(base_save_path, f'run_{run_number}', fatigue_state_name)
    
    if not os.path.exists(run_path):
        os.makedirs(run_path, exist_ok=True)
    
    # Get initial fatigue state values
    fatigue_initial, fatigue_idx_initial = fatigue_mapping[fatigue_state_name]
    
    # Run evaluation
    trajectories = evaluator.eval_single_run(fatigue_initial, fatigue_idx_initial)
    fatigue_kesav, fatigue_adp, taskload_kesav, taskload_adp = trajectories
    
    # Save trajectory data
    save_trajectory_data(run_path, fatigue_kesav, fatigue_adp, taskload_kesav, taskload_adp)


def run_comprehensive_single_evaluation(args, num_runs=10):
    """
    Run comprehensive single run evaluation across multiple runs and fatigue states.
    
    This function evaluates both ADP and Kesav algorithms starting from different
    initial fatigue states across multiple independent runs to analyze their
    trajectory behavior under various conditions.
    
    Args:
        args (argparse.Namespace): Configuration parameters
        num_runs (int): Number of independent runs to perform
        
    Returns:
        None
        
    Side Effects:
        - Creates output directories
        - Saves trajectory data for all runs and fatigue states
        
    Mathematical Context:
        Evaluates algorithm performance starting from different initial conditions:
        - Low fatigue (F_0 = 0): Well-rested operator
        - High fatigue (F_0 = 2): Fatigued operator
    """
    print(f"Starting comprehensive single run evaluation with {num_runs} runs...")
    
    # Initialize evaluator
    evaluator = Evaluations(args)
    
    # Create output directory
    save_path = 'results_single_run/'
    create_output_directory(save_path)
    
    # Get fatigue state mapping
    fatigue_mapping = get_fatigue_state_mapping()
    initial_fatigue_states = list(fatigue_mapping.keys())
    
    print(f"Evaluating {len(initial_fatigue_states)} fatigue states: {initial_fatigue_states}")
    
    # Run evaluations for all combinations
    total_evaluations = num_runs * len(initial_fatigue_states)
    
    with tqdm(total=total_evaluations, desc="Running single evaluations") as pbar:
        for run in range(1, num_runs + 1):
            for fatigue_state in initial_fatigue_states:
                # Run single evaluation
                run_single_evaluation(
                    evaluator, run, fatigue_state, save_path, fatigue_mapping
                )
                
                pbar.set_description(f"Run {run}, State: {fatigue_state}")
                pbar.update(1)
    
    print(f"Comprehensive evaluation completed. Results saved to: {save_path}")
    print(f"Total evaluations performed: {total_evaluations}")


def validate_arguments(args):
    """
    Validate argument values for consistency and correctness.
    
    Args:
        args (argparse.Namespace): Parsed arguments
        
    Raises:
        ValueError: If arguments are invalid or inconsistent
    """
    # Validate probability values
    if not (0 <= args.alpha_tp <= 1 and 0 <= args.beta_tp <= 1):
        raise ValueError("alpha_tp and beta_tp must be in range [0, 1]")
    
    if not (0 <= args.alpha_fp <= 1 and 0 <= args.beta_fp <= 1):
        raise ValueError("alpha_fp and beta_fp must be in range [0, 1]")
    
    # Validate prior probabilities
    if not (abs(sum(args.prior) - 1.0) < 1e-6):
        raise ValueError("Prior probabilities must sum to 1.0")
    
    # Validate positive parameters
    positive_params = ['num_tasks_per_batch', 'horizon', 'num_eval_runs']
    for param in positive_params:
        if getattr(args, param) <= 0:
            raise ValueError(f"{param} must be positive")
    
    # Validate cost parameters (should be non-negative)
    cost_params = [args.ctp, args.ctn, args.cfp, args.cfn, args.cr]
    if any(cost < 0 for cost in cost_params):
        raise ValueError("All cost parameters must be non-negative")


def print_configuration_summary(args, num_runs):
    """
    Print a summary of the evaluation configuration.
    
    Args:
        args (argparse.Namespace): Configuration parameters
        num_runs (int): Number of runs
    """
    print("\n" + "="*60)
    print("SINGLE RUN EVALUATION CONFIGURATION")
    print("="*60)
    print(f"Number of runs: {num_runs}")
    print(f"Time horizon: {args.horizon}")
    print(f"Tasks per batch: {args.num_tasks_per_batch}")
    print("Initial fatigue states: Low (0), High (2)")
    print(f"Prior probabilities: {args.prior}")
    print("\nFatigue Model Parameters:")
    print(f"  α_tp = {args.alpha_tp}, β_tp = {args.beta_tp}")
    print(f"  α_fp = {args.alpha_fp}, β_fp = {args.beta_fp}")
    print("\nCost Parameters:")
    print(f"  C_tp = {args.ctp}, C_tn = {args.ctn}")
    print(f"  C_fp = {args.cfp}, C_fn = {args.cfn}, C_r = {args.cr}")
    print("="*60 + "\n")


def main():
    """
    Main function to run single evaluation pipeline.
    
    This function:
    1. Parses command line arguments
    2. Validates configuration parameters
    3. Prints configuration summary
    4. Runs comprehensive single run evaluation
    5. Reports completion status
    
    Returns:
        None
    """
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    try:
        validate_arguments(args)
    except ValueError as e:
        print(f"Error: Invalid arguments - {e}")
        return
    
    # Configuration parameters
    NUM_RUNS = 10
    
    # Print configuration summary
    print_configuration_summary(args, NUM_RUNS)
    
    # Run evaluation
    try:
        run_comprehensive_single_evaluation(args, NUM_RUNS)
        print("\n" + "="*60)
        print("SINGLE RUN EVALUATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print("Trajectory data saved for analysis of algorithm behavior")
        print("under different initial fatigue conditions.")
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        print("Please check the configuration and try again.")
        return


if __name__ == "__main__":
    main()

