from typing import Callable, Optional, Union
from dphype.task import hyperparams

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

from flwr.server.strategy import Strategy


def sum_aggregate(results):
    """Aggregate results by summing the ndarrays."""
    summed = [arr.copy() for arr in results[0][0]]
    for arrs, _ in results[1:]:
        for i, arr in enumerate(arrs):
            summed[i] += arr
    return summed


class FedSum(Strategy):
    """Federated Summation strategy."""

    def __init__(
        self,
        *,
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
        epochs=100,
        eval_epochs=20,
        data_seed=42,
        dataset="ylecun/mnist",
        non_iid=False,
        alpha=0.5,
        k=-1,
        num_clients=0,
        mode="paper",
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
        self.hyperparams = hyperparams(mode)
        # self.selected_hp_idx = -1
        self.data_seed = data_seed
        self.dataset = dataset
        self.non_iid = non_iid
        self.alpha = alpha
        self.epochs = epochs
        self.eval_epochs = eval_epochs
        self.k = k
        self.votes_per_round = []
        self.num_clients = num_clients
        self.mode = mode

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

        # if self.early_stopped:
        # return []

        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        config["epochs"] = self.eval_epochs
        config["data_seed"] = self.data_seed
        config["dataset"] = self.dataset
        config["server_round"] = server_round
        config["hyperparam_id"] = server_round - 1
        config["num_clients"] = len(client_manager.all())
        config["non_iid"] = self.non_iid
        config["alpha"] = self.alpha
        config["k"] = self.k
        config["num_clients"] = self.num_clients

        print("##############################")
        print(
            self.hyperparams[server_round - 1]["lr"],
            self.hyperparams[server_round - 1]["lr_decay"],
        )

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
        # if self.early_stopped:
        # return []

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
        aggregated_ndarrays = sum_aggregate(weights_results)
        # aggregated_ndarrays[0] = aggregated_ndarrays[0] % self.M
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        self.votes_per_round.append(aggregated_ndarrays[0].item())

        if server_round == len(self.hyperparams):
            print("Voting rounds finished.")
            """
            log(
                INFO,
                f"Voting rounds finished. Best hyperparams: {self.selected_hp_idx}-> {self.hyperparams[self.selected_hp_idx]}",
            )
            """

        return parameters_aggregated, metrics_aggregated

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
