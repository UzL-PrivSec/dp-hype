import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import logging
import sys
import torch
import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fl_dp_sa.task import (
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
        self.net_initialized = False
        self.data_init = False
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
        self.device = torch.device(context.run_config["device"])
        logging.basicConfig(level=logging.DEBUG)  # or INFO, WARNING, etc.
        logging.getLogger("PIL").setLevel(logging.WARNING)
        self.hyperparams = hyperparams(self.context.run_config["mode"])

    def fit(self, parameters, config):
        if self.net_initialized == False:
            self.net = Net(config["dataset"])
            self.net_initialized = True
        set_weights(self.net, parameters)

        if self.data_init == False:
            self.trainloader, self.testloader = load_data(
                self.context,
                partition_id=self.partition_id,
                num_partitions=self.num_partitions,
                seed=config["data_seed"],
                dataset=config["dataset"],
                dirichlet=False,
                alpha=0.0,
            )
            self.data_init = True

        indices = []
        for index in range(len(self.hyperparams)):
            if f"use{index}" in config and config[f"use{index}"] == True:
                indices.append(index)

        local_rewards = np.zeros((len(self.hyperparams)))
        rets = {}
        rets["accuracy"] = 0
        if "exploration" in config and config["exploration"] == True:
            for index in indices:
                h = self.hyperparams[index]
                loss_before, _ = test(self.net, self.testloader, self.device)
                results = train(
                    self.net,
                    self.trainloader,
                    self.testloader,
                    config["local_epochs"],
                    h["lr"],
                    h["lr_decay"],
                    h["momentum"],
                    self.device,
                )
                loss_after = results["val_loss"]

                if config["epsilon"] != "inf":
                    loss_before = np.clip(loss_before, 0.0, 1.0)
                    loss_after = np.clip(loss_after, 0.0, 1.0)
                    noise_scales = {
                        "0.1": 480.5738,
                        "0.25": 206.0084,
                        "0.5": 108.4330,
                        "1.0": 57.2104,
                        "3.0": 21.1650,
                    }
                    if config["epsilon"] in noise_scales.keys():
                        noise_scale = noise_scales[config["epsilon"]]
                        loss_before += np.random.normal(0.0, noise_scale)
                        loss_after += np.random.normal(0.0, noise_scale)
                    else:
                        print("Epsilon not supported.", flush=True)
                        sys.exit(0)

                local_rewards[index] = loss_before - loss_after
        else:
            h = self.hyperparams[indices[0]]
            results = train(
                self.net,
                self.trainloader,
                self.testloader,
                config["local_epochs"],
                h["lr"],
                h["lr_decay"],
                h["momentum"],
                self.device,
            )
            rets["accuracy"] = results["val_accuracy"]
        for idx in range(len(self.hyperparams)):
            rets[f"rew{idx}"] = local_rewards[idx]
        return get_weights(self.net), len(self.trainloader.dataset), rets

    def evaluate(self, parameters, config):
        return 0, len(self.testloader.dataset), {"accuracy": 0}


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
