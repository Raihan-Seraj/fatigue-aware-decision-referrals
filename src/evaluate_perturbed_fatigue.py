from envs.fatigue_model_1_perturbed import FatigueMDPPerturbed

from run_evaluation import Evaluations
from tqdm import tqdm
import os
import json
import argparse



def create_argument_parser():
    """
    Create and configure the argument parser for perturbed fatigue evaluation.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser with all required parameters
    """
    parser = argparse.ArgumentParser(
        description="Evaluate ADP and myopic algorithms with perturbed fatigue models for robustness analysis",
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
                       help='Value of the null hypothesis')
    parser.add_argument('--H1', type=int, default=3,
                       help='Value of the alternative hypothesis')

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
                       help='Base directory to save original algorithm results')

    return parser


def create_output_directory(base_path):
    """
    Create output directory for perturbed fatigue evaluation results.
    
    Args:
        base_path (str): Base path for saving results
        
    Returns:
        str: Created directory path
        
    Side Effects:
        Creates directory if it doesn't exist
    """
    save_path = os.path.join(base_path, 'results_perturbed_fatigue')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        print(f"Created output directory: {save_path}")
    
    return save_path


def save_parameters(args, save_path):
    """
    Save configuration parameters to JSON file.
    
    Args:
        args (argparse.Namespace): Configuration parameters
        save_path (str): Directory path to save parameters
        
    Returns:
        None
        
    Side Effects:
        Creates params.json file in the specified directory
    """
    params_dict = vars(args)
    params_file = os.path.join(save_path, 'params.json')
    
    with open(params_file, 'w') as json_file:
        json.dump(params_dict, json_file, indent=4)
    
    print(f"Parameters saved to: {params_file}")


def run_perturbed_fatigue_evaluation(args, num_models=10):
    """
    Run robustness evaluation with multiple perturbed fatigue models.
    
    This function evaluates the performance of ADP and myopic algorithms
    across multiple randomly perturbed fatigue transition models to assess
    robustness to model uncertainty.
    
    Args:
        args (argparse.Namespace): Configuration parameters
        num_models (int): Number of perturbed models to evaluate
        
    Returns:
        None
        
    Side Effects:
        - Creates output directories
        - Saves configuration parameters as JSON
        - Saves evaluation results for each perturbed model
        - Saves perturbed transition matrices
        
    Mathematical Context:
        Each perturbed model modifies the original fatigue transition probabilities
        P(F_{t+1} | F_t, w_t) with random perturbations to test algorithm robustness.
    """
    print(f"Starting perturbed fatigue evaluation with {num_models} models...")
    
    # Initialize evaluator with original parameters
    evaluator = Evaluations(args)
    
    # Create base output directory
    save_path = create_output_directory(os.getcwd())
    
    # Save parameters to JSON file
    save_parameters(args, save_path)
    
    print(f"Evaluating robustness across {num_models} perturbed fatigue models...")
    
    # Evaluate each perturbed model
    for model_idx in tqdm(range(1, num_models + 1), desc="Evaluating perturbed models"):
        # Create new perturbed fatigue environment
        # Each initialization creates different random perturbations
        perturbed_env = FatigueMDPPerturbed()
        
        # Create model-specific output directory
        model_path = _create_model_directory(save_path, model_idx)
        
        # Run evaluation with perturbed model
        print(f"Evaluating model {model_idx}/{num_models}...")
        evaluator.compute_performance_perturbed_fatigue(perturbed_env, model_path)
        
        print(f"Completed evaluation for perturbed model {model_idx}")
    
    print(f"Perturbed fatigue evaluation completed. Results saved to: {save_path}")


def _create_model_directory(base_path, model_idx):
    """
    Create directory for specific perturbed model results.
    
    Args:
        base_path (str): Base directory path
        model_idx (int): Model index number
        
    Returns:
        str: Path to model-specific directory
    """
    model_path = os.path.join(base_path, f'model_{model_idx}')
    
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
    
    return model_path


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
    
    if not all(0 <= p <= 1 for p in args.prior):
        raise ValueError("Prior probabilities must be in range [0, 1]")
    
    # Validate positive integer parameters
    if args.num_tasks_per_batch <= 0:
        raise ValueError("num_tasks_per_batch must be positive")
    
    if args.horizon <= 0:
        raise ValueError("horizon must be positive")
    
    if args.num_eval_runs <= 0:
        raise ValueError("num_eval_runs must be positive")
    
    # Validate cost parameters (should be non-negative)
    cost_params = [args.ctp, args.ctn, args.cfp, args.cfn, args.cr]
    if any(cost < 0 for cost in cost_params):
        raise ValueError("All cost parameters must be non-negative")


def print_configuration_summary(args, num_models):
    """
    Print a summary of the evaluation configuration.
    
    Args:
        args (argparse.Namespace): Configuration parameters
        num_models (int): Number of perturbed models
    """
    print("\n" + "="*60)
    print("PERTURBED FATIGUE EVALUATION CONFIGURATION")
    print("="*60)
    print(f"Number of perturbed models: {num_models}")
    print(f"Monte Carlo runs per model: {args.num_eval_runs}")
    print(f"Time horizon: {args.horizon}")
    print(f"Tasks per batch: {args.num_tasks_per_batch}")
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
    Main function to run perturbed fatigue evaluation pipeline.
    
    This function:
    1. Parses command line arguments
    2. Validates configuration parameters
    3. Prints configuration summary
    4. Runs robustness evaluation across multiple perturbed models
    5. Saves configuration parameters as JSON
    6. Reports completion status
    
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
    NUM_PERTURBED_MODELS = 10
    
    # Print configuration summary
    print_configuration_summary(args, NUM_PERTURBED_MODELS)
    
    # Run perturbed fatigue evaluation
    try:
        run_perturbed_fatigue_evaluation(args, NUM_PERTURBED_MODELS)
        print("\n" + "="*60)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print("Results can be used to analyze algorithm robustness")
        print("to fatigue model uncertainties and perturbations.")
        print("Configuration parameters have been saved as params.json")
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        print("Please check the configuration and try again.")
        return


if __name__ == "__main__":
    main()