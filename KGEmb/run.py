import argparse
import json
import logging
import os
import wandb

import torch
import torch.optim

import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models
from optimizers.kg_optimizer import KGOptimizer
from utils.train import get_savedir, avg_both, format_metrics, count_params

parser = argparse.ArgumentParser(description="Knowledge Graph Embedding")
parser.add_argument(
    "--dataset", default="WN18RR",
    choices=["FB15K", "WN", "WN18RR", "FB237", "YAGO3-10", "KG_ALL", "KG_PD",
             "KG_GO", "KG_F", "KG_F_HasD_noV", "KG_F_HasD_V",
             "KG_filtered_v8", "KG_filtered_v9", "KG_filtered_v10", "KG_filtered_v11", "KG_filtered_v12", "KG_filtered_v13"],
    help="Knowledge Graph dataset"
)
parser.add_argument("--model", default="RotE", choices=all_models, help="Knowledge Graph embedding model")
parser.add_argument("--regularizer", choices=["N3", "F2"], default="N3", help="Regularizer")
parser.add_argument("--reg", default=0, type=float, help="Regularization weight")
parser.add_argument("--optimizer", choices=["Adagrad", "Adam", "SparseAdam"], default="Adagrad", help="Optimizer")
parser.add_argument("--max_epochs", default=50, type=int, help="Maximum number of epochs to train for")
parser.add_argument("--patience", default=10, type=int, help="Number of epochs before early stopping")
parser.add_argument("--valid", default=3, type=float, help="Number of epochs between validation checks")
parser.add_argument("--rank", default=1000, type=int, help="Embedding dimension")
parser.add_argument("--batch_size", default=1000, type=int, help="Batch size")
parser.add_argument("--neg_sample_size", default=50, type=int, help="Negative sample size, -1 to not use negative sampling")
parser.add_argument("--dropout", default=0, type=float, help="Dropout rate")
parser.add_argument("--init_size", default=1e-3, type=float, help="Initial embeddings' scale")
parser.add_argument("--learning_rate", default=1e-1, type=float, help="Learning rate")
parser.add_argument("--gamma", default=0, type=float, help="Margin for distance-based losses")
parser.add_argument("--bias", default="constant", choices=["constant", "learn", "none"], help="Bias type (none for no bias)")
parser.add_argument("--dtype", default="double", choices=["single", "double"], help="Machine precision")
parser.add_argument("--double_neg", action="store_true", help="Whether to negative sample both head and tail entities")
parser.add_argument("--debug", action="store_true", help="Only use 1000 examples for debugging")
parser.add_argument("--multi_c", action="store_true", help="Multiple curvatures per relation")
parser.add_argument("--no_valid", action="store_true", help="Skip loading/using a validation split; only train on train and evaluate on test.")


def train(args):
    save_dir = get_savedir(args.model, args.dataset)

    # file logger
    logging.basicConfig(
        #force=True,
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, "train.log")
    )

    # stdout logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.getLogger().setLevel(logging.INFO) #newline
    logging.info("Saving logs in: %s", save_dir)

    # create dataset, pass no_valid flag
    dataset_path = os.path.join(os.environ["DATA_PATH"], args.dataset)
    dataset = KGDataset(dataset_path, args.debug, no_valid=args.no_valid)
    args.sizes = dataset.get_shape()

    # load data
    logging.info("\tDataset shape: %s", str(dataset.get_shape()))
    train_examples = dataset.get_examples("train")
    valid_examples = None if args.no_valid else dataset.get_examples("valid")
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()

    # save config
    with open(os.path.join(save_dir, "config.json"), "w") as fjson:
        json.dump(vars(args), fjson)

    # init wandb
    run = wandb.init(
        name=f"{'||'.join(save_dir.split('/')[-3:])}||_bs{args.batch_size}_lr{args.learning_rate}_nS{args.neg_sample_size}_rk{args.rank}",
        entity='GNNEmbed',
        project='KGEmbed',
        config={**vars(args), "dataset_size": str(dataset.get_shape())}
    )

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
        args.batch_size, args.neg_sample_size, bool(args.double_neg)
    )


    # Set dynamically evaluation batch size (crucial for MurP model which uses a lot of VRAM)
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
        run.log({"epoch": step, "avg_train_loss": train_loss})

        if args.no_valid:
            logging.info("\tSaving model after last epoch since validation is disabled")
            torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
            model.cuda()
        else:
            # validation
            model.eval()
            valid_loss = optimizer.calculate_valid_loss(valid_examples)
            logging.info("\tEpoch %d | avg valid loss: %.4f", step, valid_loss)
            run.log({"avg_valid_loss": valid_loss})

            if (step + 1) % args.valid == 0:
                valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters, batch_size=eval_batch_size))
                logging.info(format_metrics(valid_metrics, split="valid"))
                run.log({
                    'MR': valid_metrics['MR'],
                    'MRR': valid_metrics['MRR'],
                    'hits@1': valid_metrics['hits@[1,3,10]'][0],
                    'hits@3': valid_metrics['hits@[1,3,10]'][1],
                    'hits@10': valid_metrics['hits@[1,3,10]'][2],
                })

                valid_mrr = valid_metrics['MRR']
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

    logging.info("\tOptimization finished")
    if best_mrr is None:
        torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
    else:
        logging.info("\tLoading best model from epoch %d", best_epoch)
        model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
    model.cuda()
    model.eval()

    if not args.no_valid:
        # final valid metrics
        valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters, batch_size=eval_batch_size))
        logging.info(format_metrics(valid_metrics, split="valid"))

    # final test metrics
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters, batch_size=eval_batch_size))
    logging.info(format_metrics(test_metrics, split="test"))


if __name__ == "__main__":
    train(parser.parse_args())
