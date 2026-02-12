import numpy as np
import optuna
from argparse import Namespace, ArgumentParser
import logging
import copy
import os

try:
    from hpo_search import run_training_pipeline
except ImportError:
    print("Error: Could not import 'run_training_pipeline'.")
    print("Please make sure 'hpo_search.py' is in the same directory.")
    exit(1)


def create_objective(rank_to_run):
    """
    Returns an objective function that has the 'rank_to_run'
    variable locked into its scope.
    """
    
    def objective(trial):
        """
        Optuna objective for a SINGLE, FIXED rank (v10).
        1 trial = 2 runs (1 AttH, 1 AttE) at the fixed rank.
        """
        
        # --- Define Search Space (i.e. the model parameters to explore)---
        base_args = Namespace(

            rank=rank_to_run, #rank is a fixed parameter
            
            # Tuned parameters 
            optimizer=trial.suggest_categorical('optimizer', ['Adagrad', 'Adam']),
            regularizer=trial.suggest_categorical('regularizer', ['N3', 'F2']),
            learning_rate=trial.suggest_float('learning_rate', 1e-5, 5e-2, log=True),
            gamma=trial.suggest_float('gamma', 0.0, 20.0), 
            reg=trial.suggest_float('reg', 0.0, 1e-3),     
            neg_sample_size=trial.suggest_int('neg_sample_size', 20, 100),
            double_neg=trial.suggest_categorical('double_neg', [True, False]),
            batch_size=trial.suggest_categorical('batch_size', [256, 512, 1000]),
            dropout=trial.suggest_float('dropout', 0.0, 0.5),
            init_size=trial.suggest_float('init_size', 1e-4, 1e-2, log=True),
            bias=trial.suggest_categorical('bias', ["constant", "learn", "none"]),
            
            # Fixed parameters 
            dataset='KG_filtered_v10', # v_10 takes a [train,valid,test] split #TODO: avoid hardcoding the dataset
            no_valid=False,
            max_epochs=32,
            patience=10,
            valid=3,
            multi_c=True,
            debug=False,
            dtype="double"
        )

        try:
            # We run AttH without pruning bad runs, because we have a multi-objective
            print(f"\n--- TRIAL {trial.number}: Running AttH (v10, rank={base_args.rank})... ---")
            args_atth = copy.deepcopy(base_args)
            args_atth.model = 'AttH'
            # trial=None disables pruning 
            metrics_atth = run_training_pipeline(args_atth, trial=None)
            atth_full_mrr = metrics_atth["full_mrr"]
            atth_full_mr = metrics_atth["full_mr"]

            # We Run for AttE (still NO Pruning) 
            print(f"\n--- TRIAL {trial.number}: Running AttE (v10, rank={base_args.rank})... ---")
            args_atte = copy.deepcopy(base_args)
            args_atte.model = 'AttE'
            metrics_atte = run_training_pipeline(args_atte, trial=None)
            atte_full_mrr = metrics_atte["full_mrr"]

        except RuntimeError as e:
            logging.error(f"Trial FAILED: {e}")
            raise optuna.exceptions.TrialPruned(f"Failed: {e}")

        # Calculate the multi-objective (GAP, ATTH MRR and ATTH MR)
        mrr_gap = atth_full_mrr - atte_full_mrr
        atth_baseline_mrr = atth_full_mrr
        atth_baseline_mr = atth_full_mr

        print(f"\n--- TRIAL {trial.number} COMPLETE (v10, rank={base_args.rank}) ---")
        print(f"  AttH FULL MRR: {atth_baseline_mrr:.6f} | MR: {atth_baseline_mr:.2f}")
        print(f"  AttE FULL MRR: {atte_full_mrr:.6f}")
        print(f"  Objective 1 (Gap): {mrr_gap:.6f}")
        print(f"  Objective 2 (AttH MRR): {atth_baseline_mrr:.6f}")
        print(f"  Objective 3 (AttH MR): {atth_baseline_mr:.2f}")

        return mrr_gap, atth_baseline_mrr, atth_baseline_mr
    
    
    return objective


if __name__ == "__main__":
    
    parser = ArgumentParser(description="Run HPO for v10 for a single fixed rank.")
    parser.add_argument("--rank", type=int, required=True, help="The fixed rank to optimize for.")
    parser.addgument("--trials", type=int, default=100, help="Number of trials to run.")
    args = parser.parse_args()
    
    
    study_name = f"KGE_v10_Comp_Rank_{args.rank}"
    storage_db = f"sqlite:///{os.path.join(os.path.dirname(os.path.abspath(__file__)), f'kge_hpo_study_v10_comp_rank_{args.rank}.db')}"
    
    print(f"--- Starting HPO for RANK={args.rank} (v10) ---")
    print(f"Study Name: {study_name}")
    print(f"Database: {storage_db}")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_db,
        load_if_exists=True,
        directions=["maximize", "maximize", "minimize"] # Max Gap, Max MRR, Min MR
    )
    
    # "HINTs"  for tuned parameters
    if len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.RUNNING))) == 0:
        print("--- Enqueuing user hint as Trial 0 ---")
        user_hint = {
            'optimizer': 'Adam',
            'reg': 0.0,
            'learning_rate': 0.0005,
            'gamma': 0.0,
            'neg_sample_size': 50,
            'double_neg': False,
            'batch_size': 1000,
            'dropout': 0.0,
            'init_size': 1e-3,
            'bias': 'constant'
        }
        study.enqueue_trial(user_hint)
    
    objective_for_this_rank = create_objective(args.rank)
    
    study.optimize(objective_for_this_rank, n_trials=args.trials)

    
    print("\n\n===== HPO STUDY COMPLETE =====")
    print(f"Study: {study_name}")
    print(f"Total trials run (all-time): {len(study.trials)}")
    
    print(f"\n--- Best Trials (Pareto Front) ---")
    if not study.best_trials:
        print("No trials completed successfully.")
    else:
        print(f"Found {len(study.best_trials)} non-dominated trials:")
        for trial in study.best_trials:
            print(f"  --- Trial #{trial.number} ---")
            print(f"    Values (Gap: {trial.values[0]:.6f}, AttH MRR: {trial.values[1]:.6f}, AttH MR: {trial.values[2]:.2f})")
            print(f"    Params: {trial.params}")
