from typing import List, Tuple
from .task import Net, get_weights, hyperparams
import numpy as np

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.workflow import DefaultWorkflow
from .fedsum import FedSum
from .fedavg import FedAvg

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

    l = [m["train_accuracy"] for num_examples, m in metrics]
    print(l)

    return {
        "train_loss": sum(train_losses) / sum(examples),
        "train_accuracy": sum(train_accuracies) / sum(examples),
        "val_loss": sum(val_losses) / sum(examples),
        "val_accuracy": sum(val_accuracies) / sum(examples),
    }


def sum_integers(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    counts = [num_examples for num_examples, _ in metrics]
    values = [m["value"] for _, m in metrics]
    total_value = sum(values)
    accuracies = [m["accuracy"] for _, m in metrics]
    avg_accuracy = sum(accuracies) / len(accuracies)

    print("###############################################################")
    print(accuracies)
    print(np.max(accuracies))
    print(np.min(accuracies))

    return {
        "sum_value": total_value,
        "num_clients": len(values),
        "avg_accuracy": avg_accuracy,
    }


def global_eval(selected_hp_idx, context, data_seed, grid):

    # Initialize global model
    model_weights = get_weights(Net(context.run_config["dataset"]))
    parameters = ndarrays_to_parameters(model_weights)

    num_sampled_clients = context.run_config["num-sampled-clients"]
    fraction_fit = 1.0
    min_fit_clients = int(num_sampled_clients * fraction_fit)

    org_strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,
        min_fit_clients=min_fit_clients,
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
        hyperparams_id=selected_hp_idx,
        data_seed=data_seed,
        dataset=context.run_config["dataset"],
        non_iid=context.run_config["non_iid"] == "yes",
        alpha=context.run_config["alpha"],
        epochs=context.run_config["local_epochs"],
    )

    legacy_context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=context.run_config["eval_rounds"]),
        strategy=org_strategy,
    )

    workflow = DefaultWorkflow()

    workflow(grid, legacy_context)

    return org_strategy.accuracy


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:

    num_sampled_clients = context.run_config["num-sampled-clients"]
    if int(num_sampled_clients) < 100:
        raise ValueError("Currently, only num-sampled-clients > 100 is supported.")

    algo_name = "dp-hype_subsets"
    eps_scenario = context.run_config["eps_scenario"]
    iidness = (
        f"dirichlet-{context.run_config['alpha']}"
        if "non_iid" in context.run_config and context.run_config["non_iid"] == "yes"
        else "iid"
    )

    dataset_name = context.run_config["dataset"]
    if "/" in dataset_name:
        dataset_name = dataset_name.replace("/", "-")

    file_path = f"./dphype/results/{context.run_config['mode']}/{algo_name}-{dataset_name}-{iidness}-N{context.run_config['num-sampled-clients']}-eps{eps_scenario}.csv"
    f = open(file_path, "w")
    f.write("eps,hp_idx,accuracy\n")
    f.flush()

    num_runs = context.run_config["num_runs"]
    run_eps = []
    run_accuracies = []
    run_hyperparams = []

    for run in range(num_runs):

        print("####################################################")
        print("run", run)

        data_seed = context.run_config["data_seed"]  # random.randint(0, 10000)

        if int(context.run_config["k"]) != 5:
            raise ValueError("Currently, only k=5 is supported.")

        # Initialize global model
        # model_weights = get_weights(Net(context.run_config['dataset']))
        parameters = ndarrays_to_parameters([np.array([0], dtype=np.int32)])

        num_sampled_clients = context.run_config["num-sampled-clients"]
        fraction_fit = 1.0
        min_fit_clients = int(num_sampled_clients * fraction_fit)

        random_client_indices = np.random.permutation(int(num_sampled_clients))
        subset_indices = np.array_split(
            random_client_indices, len(hyperparams(context.run_config["mode"]))
        )

        # Note: The fraction_fit value is configured based on the DP hyperparameter `num-sampled-clients`.
        dphype_strategy = FedSum(
            fraction_fit=fraction_fit,
            fraction_evaluate=0.0,
            min_fit_clients=min_fit_clients,
            fit_metrics_aggregation_fn=sum_integers,
            initial_parameters=parameters,
            epochs=len(hyperparams(context.run_config["mode"])),
            data_seed=data_seed,
            dataset=context.run_config["dataset"],
            non_iid=context.run_config["non_iid"] == "yes",
            alpha=context.run_config["alpha"],
            eval_epochs=context.run_config["local_epochs"],
            k=context.run_config["k"],
            num_clients=num_sampled_clients,
            subset_indices=subset_indices,
            mode=context.run_config["mode"],
        )

        # Construct the LegacyContext
        legacy_context = LegacyContext(
            context=context,
            config=ServerConfig(len(hyperparams(context.run_config["mode"]))),
            strategy=dphype_strategy,
        )

        # Create the train/evaluate workflow
        workflow = DefaultWorkflow()

        # Execute
        workflow(grid, legacy_context)

        if eps_scenario == "All":
            if context.run_config["mode"] == "dry":
                noise_scale_by_eps = {
                    "Inf": 0.0,
                }
            else:
                noise_scale_by_eps = {
                    "0.1": 45.0,
                    "0.25": 25.0,
                    "0.5": 10.7,
                    "1.0": 5.5,
                    "3.0": 2.08,
                    "Inf": 0.0,
                }
        else:
            raise ValueError("Unknown epsilon scenario")

        votes_per_round = np.array(dphype_strategy.votes_per_round) / 100

        print(votes_per_round)

        subset_sizes = np.array([len(subset) for subset in subset_indices])
        votes_per_round = votes_per_round / subset_sizes

        print(votes_per_round)

        print("Starting to evaluate privacy budgets")

        for eps, noise_scale in noise_scale_by_eps.items():

            print("## Eps:", eps, "-> noise scale:", noise_scale)

            noisy_votes_per_round = votes_per_round + np.random.normal(
                0, noise_scale, len(votes_per_round)
            )
            print(noisy_votes_per_round)
            selected_hp_idx = int(np.argmax(noisy_votes_per_round))

            acc = global_eval(selected_hp_idx, context, data_seed, grid)

            run_eps.append(eps)
            run_accuracies.append(acc)
            run_hyperparams.append(selected_hp_idx)
            f.write(f"{eps},{selected_hp_idx},{acc}\n")
            f.flush()

    print(run_eps)
    print(run_hyperparams)
    print(run_accuracies)
    f.flush()
    f.close()
