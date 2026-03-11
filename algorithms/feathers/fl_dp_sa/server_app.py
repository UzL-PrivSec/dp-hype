from typing import List, Tuple
from fl_dp_sa.task import Net, get_weights, hyperparams
import random
import numpy as np
import csv
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.workflow import DefaultWorkflow
from .fedsum import FedSum

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    examples = [num_examples for num_examples, _ in metrics]
    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    train_accuracies = [
        num_examples * m["train_accuracy"] for num_examples, m in metrics
    ]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]

    return {
        "train_loss": sum(train_losses) / sum(examples),
        "train_accuracy": sum(train_accuracies) / sum(examples),
        "val_loss": sum(val_losses) / sum(examples),
        "val_accuracy": sum(val_accuracies) / sum(examples),
    }


def sum_integers(mode, metrics: List[Tuple[int, Metrics]]) -> Metrics:
    rets = {}
    examples = [num_examples for num_examples, _ in metrics]
    val_accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    rets["accuracy"] = sum(val_accuracies) / sum(examples)
    # For debugging, here you could print the rewards per client
    # print('#####################################')
    for idx in range(len(hyperparams(mode))):
        rets[f"rew{idx}"] = sum(
            np.nan_to_num([m[f"rew{idx}"] for _, m in metrics], nan=0.0)
        ) / len(metrics)
        l = [m[f"rew{idx}"] for _, m in metrics]
        # print the returned rewards (one reward for each client) for this hyperparameter
        # print(l)

    return rets


def load_accuracy_by_id(csv_path):
    """Read a CSV and return a dictionary mapping id to accuracy."""
    accuracy_by_id = {}
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            id_ = int(row["id"])
            accuracy = float(row["accuracy"])
            accuracy_by_id[id_] = accuracy
    return accuracy_by_id


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:

    algo_name = "feathers"
    eps = context.run_config["epsilon"]

    dataset_name = context.run_config["dataset"]
    if "/" in dataset_name:
        dataset_name = dataset_name.replace("/", "-")

    file_path = f"./fl_dp_sa/results/{context.run_config['mode']}/{algo_name}-{dataset_name}-N{context.run_config['num-sampled-clients']}-eps{eps}.csv"
    if context.run_config["mode"] == "artifacts":
        if os.path.exists(file_path):
            f = open(file_path, "a")
        else:
            f = open(file_path, "w")
            f.write("accuracy\n")
            f.flush()
    else:
        f = open(file_path, "w")
        f.write("accuracy\n")
        f.flush()

    num_runs = context.run_config["num_runs"]
    results = []
    for run in range(num_runs):

        print("####################################################")
        print("run", run)

        model_weights = get_weights(Net(context.run_config["dataset"]))
        parameters = ndarrays_to_parameters(model_weights)

        num_sampled_clients = context.run_config["num-sampled-clients"]
        fraction_fit = 1.0
        min_fit_clients = int(num_sampled_clients * fraction_fit)

        data_seed = random.randint(0, 10000)

        feathers_strategy = FedSum(
            fraction_fit=fraction_fit,
            fraction_evaluate=0.0,
            min_fit_clients=min_fit_clients,
            fit_metrics_aggregation_fn=lambda args: sum_integers(
                context.run_config["mode"], args
            ),
            initial_parameters=parameters,
            data_seed=data_seed,
            dataset=context.run_config["dataset"],
            gamma=context.run_config["gamma"],
            alpha=-1,
            max_exploration_steps=context.run_config["max_exploration_steps"],
            local_epochs=context.run_config["local_epochs"],
            epsilon=context.run_config["epsilon"],
            mode=context.run_config["mode"],
        )

        legacy_context = LegacyContext(
            context=context,
            config=ServerConfig(num_rounds=context.run_config["rounds"]),
            strategy=feathers_strategy,
        )

        workflow = DefaultWorkflow()

        workflow(grid, legacy_context)

        results.append(feathers_strategy.accuracy)

        f.write(f"{feathers_strategy.accuracy}\n")
        f.flush()

        print("####################################################")
        print("results so far")
        print(results)

    f.flush()
    f.close()
