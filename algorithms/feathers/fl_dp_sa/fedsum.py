from typing import Callable, Optional, Union
from fl_dp_sa.task import hyperparams

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate
from flwr.server.strategy import Strategy


from fl_dp_sa.task import softmax, update_rewards
import numpy as np
from scipy.stats import entropy


def sum_aggregate(results):
    """Aggregate results by summing the ndarrays."""

    summed = [arr.copy() for arr in results[0][0]]
    for arrs, _ in results[1:]:
        for i, arr in enumerate(arrs):
            print(arr)
            summed[i] += arr
    return summed


class FedSum(Strategy):
    """Federated Summation strategy."""

    def __init__(
        self,
        *,
        alpha,
        gamma,
        max_exploration_steps,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        data_seed=42,
        dataset="ylecun/mnist",
        local_epochs=5,
        epsilon=1.0,
        mode="",
    ) -> None:
        super().__init__()

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.early_stopped = False
        self.early_stopped_hp_id = -1
        self.M = 2**32
        self.data_seed = data_seed
        self.dataset = dataset

        self.mode = mode
        self.hyperparams = hyperparams(mode)
        self.r = np.zeros(len(self.hyperparams))
        self.P = softmax(self.r)
        self.sampled_indices = []
        self.alpha = alpha
        self.gamma = gamma
        self.max_exploration_steps = max_exploration_steps
        self.overall_exploration_steps = 0
        self.accuracy = 0
        self.local_epochs = local_epochs
        self.epsilon = epsilon

    def __repr__(self) -> str:
        return f"FedSum(accept_failures={self.accept_failures})"

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        initial_parameters = self.initial_parameters
        self.initial_parameters = None
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        if self.evaluate_fn is None:
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:

        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        if server_round % 10 == 1:
            if not np.all(self.r == 0):
                normed_rewards = self.r / np.linalg.norm(self.r, float("inf"))
            else:
                normed_rewards = self.r
            dist = softmax(normed_rewards)
            exploration_steps = int(np.round(self.gamma * entropy(dist), 0))
            self.overall_exploration_steps += exploration_steps
            if (
                self.overall_exploration_steps > self.max_exploration_steps
                and self.epsilon != "inf"
            ):
                exploration_steps -= (
                    self.overall_exploration_steps - self.max_exploration_steps
                )
                self.overall_exploration_steps = self.max_exploration_steps
            self.sampled_indices = np.random.choice(
                len(self.hyperparams), size=exploration_steps, p=self.P
            )

            print("########################################################")
            print(self.sampled_indices, flush=True)
            for index in self.sampled_indices:
                config[f"use{index}"] = True
            config["exploration"] = True
        else:
            use_idx = np.argmax(self.r)
            config[f"use{use_idx}"] = True
            print("########################################################")
            print(self.hyperparams[use_idx], flush=True)
        config["dataset"] = self.dataset
        config["data_seed"] = self.data_seed
        config["local_epochs"] = self.local_epochs
        config["epsilon"] = self.epsilon

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        return [
            (client, FitIns(parameters, config)) for idx, client in enumerate(clients)
        ]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        if self.early_stopped:
            return []

        if self.fraction_evaluate == 0.0:
            return []

        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)

        evaluate_ins = EvaluateIns(parameters, config)

        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_ndarrays = aggregate(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        if server_round % 10 == 1:
            r_e = np.zeros((len(hyperparams(self.mode))))
            for idx in range(len(hyperparams(self.mode))):
                r_e[idx] = metrics_aggregated[f"rew{idx}"]

            print("###############################")
            print("rewards as update")
            print([(idx, p) for idx, p in enumerate(r_e)], flush=True)

            i = np.array(
                [
                    1 if idx in self.sampled_indices else 0
                    for idx in range(len(self.hyperparams))
                ]
            )
            self.r = update_rewards(self.r, r_e, i, alpha=self.alpha)
            self.P = softmax(self.r)

        self.accuracy = metrics_aggregated["accuracy"]

        print("########################################################")
        print("rewards")
        print([(idx, p) for idx, p in enumerate(self.r)])
        print("distribution")
        print([(idx, p) for idx, p in enumerate(np.round(self.P, 3))])
        print("best hp")
        print(self.hyperparams[np.argmax(self.r)])
        print("accuracy")
        print(metrics_aggregated["accuracy"], flush=True)

        return (
            None if server_round % 10 == 1 else parameters_aggregated
        ), metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        total_loss = sum(
            evaluate_res.loss for _, evaluate_res in results
        )  # No weighting here either (sum)
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)

        return total_loss, metrics_aggregated
