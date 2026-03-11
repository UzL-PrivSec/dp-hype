import flwr as fl
from flwr.client import NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.server.strategy import FedAvg
from flwr.server.workflow import SecAggPlusWorkflow, DefaultWorkflow
from flwr.server import ServerApp, ServerConfig, LegacyContext, Grid
from flwr.common import Context, Message
from typing import Optional
from flwr.common.logger import FLOWER_LOGGER
import csv
import sys
import time

import argparse
import numpy as np


class LoggingGrid:

    def __init__(self, wrapped_grid: Grid):
        self._wrapped_grid = wrapped_grid
        self.client_stats = {}  # {node_id: {'sent': bytes, 'received': bytes}}

    def __getattr__(self, name):
        return getattr(self._wrapped_grid, name)

    def send(self, message: Message, *args, **kwargs) -> Optional[Message]:
        node_id = getattr(message.metadata, "dst_node_id", "unknown")

        try:
            msg_size = self._estimate_message_size(message)
            if node_id not in self.client_stats:
                self.client_stats[node_id] = {"sent": 0, "received": 0}
            self.client_stats[node_id]["sent"] += msg_size
        except Exception as e:
            print(e)

        return self._wrapped_grid.send(message, *args, **kwargs)

    def receive(self, *args, **kwargs) -> Optional[Message]:
        result = self._wrapped_grid.receive(*args, **kwargs)

        if result is not None:
            node_id = getattr(result.metadata, "src_node_id", "unknown")

            try:
                msg_size = self._estimate_message_size(result)
                if node_id not in self.client_stats:
                    self.client_stats[node_id] = {"sent": 0, "received": 0}
                self.client_stats[node_id]["received"] += msg_size
            except Exception as e:
                print(e)

        return result

    def send_and_receive(self, messages, *args, **kwargs):
        is_list = isinstance(messages, list)
        msg_list = messages if is_list else [messages]

        for msg in msg_list:

            if msg.metadata.message_type in ["evaluate", "get_parameters"]:
                continue

            node_id = getattr(msg.metadata, "dst_node_id", "unknown")

            try:
                msg_size = self._estimate_message_size(msg)
                if node_id not in self.client_stats:
                    self.client_stats[node_id] = {"sent": 0, "received": 0}
                self.client_stats[node_id]["sent"] += msg_size

            except Exception as e:
                print(e)

        result = self._wrapped_grid.send_and_receive(messages, *args, **kwargs)

        if result is not None:
            result_list = result if isinstance(result, list) else [result]
            for res in result_list:
                if res is not None:

                    if res.metadata.message_type in ["evaluate", "get_parameters"]:
                        continue

                    node_id = getattr(res.metadata, "src_node_id", "unknown")

                    try:
                        msg_size = self._estimate_message_size(res)

                        if node_id not in self.client_stats:
                            self.client_stats[node_id] = {"sent": 0, "received": 0}
                        self.client_stats[node_id]["received"] += msg_size
                    except Exception as e:
                        print(e)

        return result

    def _estimate_message_size(self, message: Message) -> int:
        size = 0

        metadata_str = str(message.metadata)
        size += sys.getsizeof(metadata_str)

        if hasattr(message.content, "parameters") and message.content.parameters:
            params = message.content.parameters
            if hasattr(params, "tensors"):
                tensor_sizes = []
                for tensor in params.tensors:
                    tensor_size = len(tensor)
                    tensor_sizes.append(tensor_size)
                    size += tensor_size
                if tensor_sizes:
                    pass
            if hasattr(params, "tensor_type"):
                size += sys.getsizeof(params.tensor_type)

        if hasattr(message.content, "configs") and message.content.configs:
            config_size = sys.getsizeof(str(message.content.configs))
            size += config_size

        if (
            hasattr(message.content, "config_records")
            and message.content.config_records
        ):
            config_rec_size = sys.getsizeof(str(message.content.config_records))
            size += config_rec_size

        content_str = str(message.content)
        if len(content_str) > 100:
            size += sys.getsizeof(content_str)

        return size


vector_length = 100
num_clients_list = [50, 100, 250]
num_shares = 0.25
reconstruction_threshold = 1.0
num_runs = 5

simulation_results = {}


class FlowerClient(NumPyClient):

    def __init__(self, vector_length):
        self.vector = np.ones(vector_length)

    def get_parameters(self, config):
        return [self.vector]

    def fit(self, parameters, config):
        return [self.vector], 1, {}

    def evaluate(self, parameters, config):
        return 0.0, 1, {"accuracy": 1.0}


def client_fn(context):
    return FlowerClient(vector_length).to_client()


def create_client_app():
    mods = [secaggplus_mod]
    return fl.client.ClientApp(
        client_fn=client_fn,
        mods=mods,
    )


def create_server_app():
    server_app = ServerApp()

    @server_app.main()
    def main(grid: Grid, context: Context) -> None:
        logging_grid = LoggingGrid(grid)
        strategy = FedAvg()

        legacy_context = LegacyContext(
            context=context,
            config=ServerConfig(num_rounds=1),
            strategy=strategy,
        )

        workflow = DefaultWorkflow(
            fit_workflow=SecAggPlusWorkflow(
                num_shares=num_shares,
                reconstruction_threshold=reconstruction_threshold,
            )
        )

        workflow(logging_grid, legacy_context)

        global simulation_results
        if logging_grid.client_stats:
            num_clients_actual = len(logging_grid.client_stats)
            avg_sent = (
                sum(s["sent"] for s in logging_grid.client_stats.values())
                / num_clients_actual
            )
            avg_received = (
                sum(s["received"] for s in logging_grid.client_stats.values())
                / num_clients_actual
            )

            simulation_results = {
                "avg_upload": avg_sent,
                "avg_download": avg_received,
            }

            FLOWER_LOGGER.info("=" * 80)
            FLOWER_LOGGER.info("AVERAGE PER-CLIENT STATISTICS:")
            FLOWER_LOGGER.info(f"  Number of clients: {num_clients_actual}")
            FLOWER_LOGGER.info(f"  Avg sent per client:     {avg_sent:,.2f} bytes")
            FLOWER_LOGGER.info(f"  Avg received per client: {avg_received:,.2f} bytes")
            FLOWER_LOGGER.info("=" * 80)

    return server_app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["dry", "artifacts", "paper"],
        required=True,
    )
    args = parser.parse_args()

    if args.mode == "dry":
        num_runs = 1
    elif args.mode == "artifacts":
        num_runs = 5
    elif args.mode == "paper":
        num_runs = 5

    csv_filename = f"results/{args.mode}/bandwidth_results.csv"

    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "num_clients",
                "run",
                "avg_upload_bytes",
                "avg_download_bytes",
                "runtime_seconds",
            ]
        )

    FLOWER_LOGGER.info(f"Starting experiments. Results will be saved to {csv_filename}")

    for num_clients in num_clients_list:
        FLOWER_LOGGER.info(f"\n{'='*80}")
        FLOWER_LOGGER.info(f"STARTING EXPERIMENTS WITH {num_clients} CLIENTS")
        FLOWER_LOGGER.info(f"{'='*80}\n")

        FLOWER_LOGGER.info(f"\n{'-'*80}")
        FLOWER_LOGGER.info(f"Configuration: {num_clients} clients")
        FLOWER_LOGGER.info(f"{'-'*80}\n")

        for run in range(1, num_runs + 1):
            FLOWER_LOGGER.info(f"\n>>> Run {run}/{num_runs} with {num_clients} clients")

            try:
                simulation_results.clear()

                client_app = create_client_app()
                server_app = create_server_app()

                start_time = time.time()

                fl.simulation.run_simulation(
                    client_app=client_app,
                    server_app=server_app,
                    num_supernodes=num_clients,
                    backend_config={
                        "client_resources": {"num_cpus": 1},
                    },
                )

                end_time = time.time()
                runtime = end_time - start_time

                if simulation_results:
                    avg_upload = simulation_results.get("avg_upload", 0)
                    avg_download = simulation_results.get("avg_download", 0)

                    with open(csv_filename, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(
                            [
                                num_clients,
                                run,
                                avg_upload,
                                avg_download,
                                runtime,
                            ]
                        )

                    FLOWER_LOGGER.info(
                        f"Run {run} completed - Upload: {avg_upload:.2f}, Download: {avg_download:.2f}, Runtime: {runtime:.2f}s"
                    )
                else:
                    FLOWER_LOGGER.warning(
                        f"Run {run} completed but no results captured"
                    )

            except Exception as e:
                FLOWER_LOGGER.error(f"Run {run} failed: {e}")
                import traceback

                traceback.print_exc()

                with open(csv_filename, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        [
                            num_clients,
                            run,
                            "ERROR",
                            "ERROR",
                            "ERROR",
                        ]
                    )

    FLOWER_LOGGER.info(f"\n{'='*80}")
    FLOWER_LOGGER.info(f"All experiments completed. Results saved to {csv_filename}")
    FLOWER_LOGGER.info(f"{'='*80}\n")
