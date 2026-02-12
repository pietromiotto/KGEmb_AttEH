import optuna
import sys

def analyze_study(db_file):
    """
    Loads a study from a database and prints the best trial(s).
    """
    
    # Load the study from the database file
    storage_name = f"sqlite:///{db_file}"
    try:
        
        study = optuna.load_study(study_name=None, storage=storage_name) #None = auto detection of the study given a db name
    except KeyError:
        print(f"Error: Could not find any studies in {db_file}.")
        print("This might happen if the study failed to start.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred loading the study: {e}")
        sys.exit(1)

    print(f"--- Analysis for {db_file} ---")
    print(f"Total trials: {len(study.trials)}")
    print(f"Study direction(s): {study.directions}")

    # Check if any trials are complete
    completed_trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    if not completed_trials:
        print("\n--- No completed trials found ---")
        print("The study may have been interrupted or failed before any trial could finish.")
        return

    print("\n--- Best Trial(s) ---")
    
    num_objectives = len(study.directions)

    if not study.best_trials:
        print("No trials completed successfully.")
    else:
        print(f"Found {len(study.best_trials)} best trial(s) on the Pareto front:")
        
        for trial in study.best_trials:
            print(f"  --- Trial #{trial.number} ---")
            
            # --- This logic handles all 3 of your script types ---
            
            if num_objectives == 3: # v10 comparative
                print(f"    Values (Gap: {trial.values[0]:.6f}, AttH MRR: {trial.values[1]:.6f}, AttH MR: {trial.values[2]:.2f})")
            
            elif num_objectives == 2: # v12 comparative
                print(f"    Values (Gap: {trial.values[0]:.6f}, AttH MRR: {trial.values[1]:.6f})")
            
            elif num_objectives == 1: # AttH-only script
                print(f"    Value (RHS-MRR): {trial.values[0]:.6f}")
            
            else: # Fallback
                print(f"    Values: {trial.values}")
                
            print(f"    Params: {trial.params}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Please provide the path to the database file.")
        print("Usage: python analyze_study.py <your_database.db>")
        sys.exit(1)

    analyze_study(sys.argv[1])
