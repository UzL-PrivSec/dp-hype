import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import logging
import torch
import numpy as np
from tqdm import tqdm
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ConfigRecord

from dphype.task import (
    load_data,
    test,
    train,
    Net,
    set_weights,
    get_weights,
    hyperparams,
)


class FlowerClient(NumPyClient):
    def __init__(self, context, partition_id, num_partitions) -> None:
        self.context = context
        self.state = context.state
        if "eval_metrics" not in self.state.config_records:
            self.state.config_records["eval_metrics"] = ConfigRecord()
        self.net_initialized = False
        self.data_init = False
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
        self.device = torch.device(context.run_config["device"])
        self.hyperparams = hyperparams(context.run_config["mode"])
        logging.basicConfig(level=logging.DEBUG)  # or INFO, WARNING, etc.
        logging.getLogger("PIL").setLevel(logging.WARNING)

    def prepare_vote(self, hyperparams, config):

        print("################################################################")

        eval_metrics = self.state.config_records.setdefault("eval_metrics", {})
        eval_metrics.setdefault("all_accuracies", [])

        k = config["k"]

        scores = []  # (accuracy, idx, hyperparam_dict)

        for idx, h in tqdm(enumerate(hyperparams)):

            self.net = Net(config["dataset"])
            results = train(
                self.net,
                self.trainloader,
                self.testloader,
                config["epochs"],
                self.hyperparams[idx]["lr"],
                self.hyperparams[idx]["lr_decay"],
                self.hyperparams[idx]["momentum"],
                self.device,
            )

            acc = results["val_accuracy"]

            eval_metrics["all_accuracies"].append(acc)

            scores.append((acc, idx, h))

        scores.sort(key=lambda x: x[0], reverse=True)
        topk = scores[:k]

        eval_metrics["best_idx"] = [tpk[1] for tpk in topk]

    def fit(self, parameters, config):
        if self.net_initialized == False:
            self.net_initialized = True
            self.net = Net(config["dataset"])

        if self.data_init == False:
            self.trainloader, self.testloader = load_data(
                self.context,
                partition_id=self.partition_id,
                num_partitions=self.num_partitions,
                seed=config["data_seed"],
                dataset=config["dataset"],
                dirichlet=config["non_iid"],
                alpha=config["alpha"],
            )
            self.data_init = True

        if "evaluate" in config and config["evaluate"] == True:
            set_weights(self.net, parameters)
            _, val_acc = test(self.net, self.testloader, self.device)

            results = train(
                self.net,
                self.trainloader,
                self.testloader,
                config["epochs"],
                self.hyperparams[config["hyperparam_id"]]["lr"],
                self.hyperparams[config["hyperparam_id"]]["lr_decay"],
                self.hyperparams[config["hyperparam_id"]]["momentum"],
                self.device,
            )
            results["val_accuracy"] = val_acc

            return get_weights(self.net), len(self.trainloader.dataset), results

        else:
            if config["server_round"] == 1:
                self.prepare_vote(self.hyperparams, config)

            acc = self.state.config_records["eval_metrics"]["all_accuracies"][
                config["server_round"] - 1
            ]

            if (
                config["server_round"] - 1
                in self.state.config_records["eval_metrics"]["best_idx"]
            ):
                u = int(acc * 100)  # 1 -> convert acc into %
            else:
                u = 0

            return (
                [np.array([u])],
                len(self.trainloader.dataset),
                {"value": u, "accuracy": acc},
            )

    def evaluate(self, parameters, config):
        return (
            0,
            len(self.testloader.dataset),
            {"accuracy": 0},
        )


best_idx = {}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    client = FlowerClient(context, partition_id, num_partitions)
    return client.to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[],
)
