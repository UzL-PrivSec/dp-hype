from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode", type=str, required=True, choices=["dry", "artifacts", "paper"]
)

args = parser.parse_args()

dataset_name_mapping = {
    "scikit-learn/adult-census-income": "Adult",
    "ylecun/mnist": "MNIST",
    "uoft-cs/cifar10": "CIFAR-10",
}

for dataset in [
    "scikit-learn/adult-census-income",
    "ylecun/mnist",
    "uoft-cs/cifar10",
]:
    for alpha in [0.5, 5.0, 30.0]:

        fds = FederatedDataset(
            dataset=dataset,
            partitioners={
                "train": DirichletPartitioner(
                    num_partitions=50,
                    partition_by=(
                        "income"
                        if dataset == "scikit-learn/adult-census-income"
                        else "label"
                    ),
                    alpha=alpha,
                    seed=43,
                    min_partition_size=10,
                ),
            },
        )

        partitioner = fds.partitioners["train"]

        title_beta = (
            r"$ \beta = 0.5 $ (highly non-iid) "
            if alpha == 0.5
            else r"$ \beta = 5.0 $" if alpha == 5.0 else r"$ \beta = 30.0 $"
        )

        import matplotlib.pyplot as plt

        plt.rcParams["figure.dpi"] = 200
        plt.rcParams["text.usetex"] = False
        plt.rcParams["axes.titlesize"] = 24
        plt.rcParams["axes.labelsize"] = 24
        plt.rcParams["xtick.labelsize"] = 24
        plt.rcParams["ytick.labelsize"] = 24
        plt.rcParams["axes.titlepad"] = 20
        fig, ax, df = plot_label_distributions(
            partitioner,
            label_name=(
                "income" if dataset == "scikit-learn/adult-census-income" else "label"
            ),
            plot_type="bar",
            size_unit="absolute",
            partition_id_axis="x",
            legend=False,
            verbose_labels=True,
            title=f"{dataset_name_mapping[dataset]}_{title_beta}",
            max_num_partitions=10,
            legend_kwargs={"ncol": 10, "bbox_to_anchor": (0.5, 1.15)},
        )

        fig_prefix = (
            "figure7" if dataset == "scikit-learn/adult-census-income" else "figure6"
        )

        fig.savefig(
            f"/workspace/dp-hype/figures/{args.mode}/{fig_prefix}_dirichlet_distribution_{dataset_name_mapping[dataset]}_beta-{alpha}.pdf",
            bbox_inches="tight",
        )
