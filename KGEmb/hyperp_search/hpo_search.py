import argparse
import json
import logging
import os
import wandb #keep!

import torch
import torch.optim

import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models
from optimizers.kg_optimizer import KGOptimizer
#REPO metrics for both AttH and AttE
from utils.train import get_savedir, avg_both, format_metrics, count_params, avg_both_rhs


import optuna

def run_training_pipeline(args, trial=None):
    """
    Main training and evaluation pipeline.
    This is a modified version of your original 'train' function.
    It accepts an 'args' Namespace and an optional 'trial' object for pruning.
    
    Returns a dictionary with *all* final test metrics.
    """
    
    # Disable W&B logging for HPO trials
    os.environ["WANDB_MODE"] = "disabled" 

    save_dir = get_savedir(args.model, args.dataset)


    logging.basicConfig(
        force=True, # refresh logging config at each trial 
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, "train.log")
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    

    logger = logging.getLogger("")
    #if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(console) #(prev. under the if)
    logger.setLevel(logging.INFO)
    
    logging.info(f"Saving logs in: {save_dir}")
    
    #log hyperparameters
    logging.info(f"--- STARTING TRIAL WITH PARAMS ---")
    logging.info(json.dumps(vars(args), indent=2))

    # create dataset, passes no_valid flag
    dataset_path = os.path.join(os.environ["DATA_PATH"], args.dataset)
    dataset = KGDataset(dataset_path, args.debug, no_valid=args.no_valid)
    args.sizes = dataset.get_shape()

    # load data
    logging.info("\tDataset shape: %s", str(dataset.get_shape()))
    train_examples = dataset.get_examples("train")
    valid_examples = None if args.no_valid else dataset.get_examples("valid")
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()

    # save config file for this trial
    with open(os.path.join(save_dir, "config.json"), "w") as fjson:
        json.dump(vars(args), fjson)

    # create model
    model = getattr(models, args.model)(args)
    total = count_params(model)
    logging.info("Total number of parameters %d", total)
    device = "cuda"
    model.to(device)

    # optimizer
    regularizer = getattr(regularizers, args.regularizer)(args.reg)
    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    optimizer = KGOptimizer(
        model, regularizer, optim_method,
        args.batch_size, args.neg_sample_size, bool(args.double_neg), verbose=False)
    

    # N.W. MurP is too complex so we force smaller eval batch sizes
    eval_batch_size = 500
    if args.model == "MurP":
        if args.rank <= 32:
            eval_batch_size = 100
            logging.info(f"MurP model with small rank ({args.rank}) detected. Setting eval batch size to {eval_batch_size}.")
        else: # rank > 32
            eval_batch_size = 10 
            logging.info(f"MurP model with large rank ({args.rank}) detected. Setting eval batch size to {eval_batch_size}.")
    else:
         logging.info(f"Using default eval batch size {eval_batch_size} for model {args.model}.")

    
    counter = 0
    best_mrr = None
    best_epoch = None
    logging.info("\tStart training")

    for step in range(args.max_epochs):
        # train
        model.train()
        train_loss = optimizer.epoch(train_examples)
        logging.info("\tEpoch %d | avg train loss: %.4f", step, train_loss)

        if args.no_valid:
            # N.W. If no validation, we can't do early stopping or pruning!
            pass
        else:
            # validation
            model.eval()
            valid_loss = optimizer.calculate_valid_loss(valid_examples)
            logging.info("\tEpoch %d | avg valid loss: %.4f", step, valid_loss)

            if (step + 1) % args.valid == 0:
                valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters, batch_size=eval_batch_size))
                logging.info(format_metrics(valid_metrics, split="valid"))

                valid_mrr = valid_metrics['MRR']
                
                # PRUNING---
                if trial is not None and not args.no_valid: # Only prune if trial is passed AND we have a validation set
                    trial.report(valid_mrr, step)
                    if trial.should_prune():
                        logging.info(f"--- Trial pruned at step {step} ---")
                        raise optuna.exceptions.TrialPruned()
                #----

                # early stopping logic (if we have a validation set)
                if best_mrr is None or valid_mrr > best_mrr:
                    best_mrr = valid_mrr
                    counter = 0
                    best_epoch = step
                    logging.info("\tSaving model at epoch %d", step)
                    torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
                    model.cuda()
                else:
                    counter += 1
                    if counter == args.patience:
                        logging.info("\tEarly stopping")
                        break

    if args.no_valid:
        logging.info(f"\tValidation disabled. Saving model after final epoch {args.max_epochs}")
        torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
        model.cuda()

    logging.info("\tOptimization finished")
    if not args.no_valid:
        if best_mrr is None:
            logging.info("\tNo best model found, saving final model.")
            torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
        else:
            logging.info(f"\tLoading best model from epoch {best_epoch}")
            model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
    
    model.cuda()
    model.eval()

    if not args.no_valid:
        valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters, batch_size=eval_batch_size))
        logging.info(format_metrics(valid_metrics, split="valid"))
    
    logging.info("--- Computing RHS-only test metrics ---")
    rhs_test_metrics = avg_both_rhs(*model.compute_metrics(test_examples, filters, batch_size=eval_batch_size))
    logging.info(format_metrics(rhs_test_metrics, split="test_rhs_only"))
    
    logging.info("--- Computing FULL test metrics ---")
    full_test_metrics = avg_both(*model.compute_metrics(test_examples, filters, batch_size=eval_batch_size))
    logging.info(format_metrics(full_test_metrics, split="test"))

    # dictionary with all metrics
    return {
        "rhs_mrr": rhs_test_metrics['MRR'],
        "rhs_mr": rhs_test_metrics['MR'],
        "full_mrr": full_test_metrics['MRR'],
        "full_mr": full_test_metrics['MR']
    }
 
