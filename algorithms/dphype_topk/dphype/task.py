"""fl_dp_sa: Flower Example using Differential Privacy and Secure Aggregation."""

from collections import OrderedDict

import itertools
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm


import pickle

_ADULT_SCHEMA_PATH = "./data/adult_schema.pkl"

_ADULT_NUM_COLS = [
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
]
_ADULT_CAT_COLS = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
]


def _materialize_df(ds_split, max_rows=None):
    n = len(ds_split)
    if max_rows is not None:
        n = min(n, max_rows)
    rows = [ds_split[i] for i in range(n)]
    return pd.DataFrame(rows)


def _compute_and_cache_adult_schema_from_split(train_split):
    os.makedirs(os.path.dirname(_ADULT_SCHEMA_PATH), exist_ok=True)
    df = _materialize_df(train_split)  # only from your already-loaded split

    if "income" not in df.columns:
        raise RuntimeError(f"Adult: label 'income' not in columns {list(df.columns)}")

    num_cols = [c for c in _ADULT_NUM_COLS if c in df.columns]
    cat_cols = [c for c in _ADULT_CAT_COLS if c in df.columns]

    categories = []
    for c in cat_cols:
        cats = pd.Series(df[c]).astype(str).str.strip().dropna().unique().tolist()
        categories.append(sorted(cats))

    schema = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "categories": categories,  # aligned with cat_cols
        "label_col": "income",
    }
    with open(_ADULT_SCHEMA_PATH, "wb") as f:
        pickle.dump(schema, f)
    return schema


def _load_adult_schema():
    with open(_ADULT_SCHEMA_PATH, "rb") as f:
        return pickle.load(f)


def _adult_compute_input_dim():
    schema = _load_adult_schema()
    return len(schema["num_cols"]) + sum(len(lst) for lst in schema["categories"])


# ------------------------------------------------------------------------------------------------

import sklearn
from packaging import version

_OHE_KW = {"handle_unknown": "ignore"}
if version.parse(sklearn.__version__) >= version.parse("1.2"):
    _OHE_KW["sparse_output"] = False
else:
    _OHE_KW["sparse"] = False

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_adult_preprocessor_from_schema(schema, train_split):
    num_cols = schema["num_cols"]
    cat_cols = schema["cat_cols"]
    categories = schema["categories"]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=True, with_std=True), num_cols),
            ("cat", OneHotEncoder(categories=categories, **_OHE_KW), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    df_train = _materialize_df(train_split)
    X_train = df_train[num_cols + cat_cols]
    pre.fit(X_train)
    return pre


class Net(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

        if self.dataset == "ylecun/mnist":
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        elif self.dataset == "uoft-cs/cifar10":
            self.block1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # -> 8x8
                nn.Dropout(0.25),
            )
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(128, 10)
        elif self.dataset == "scikit-learn/adult-census-income":
            input_dim = _adult_compute_input_dim()
            self.tab1 = nn.Linear(input_dim, 64)
            self.tab2 = nn.Linear(64, 64)
            self.tab_drop = nn.Dropout(0.2)
            self.tab_out = nn.Linear(64, 1)

    def forward(self, x):
        if self.dataset == "ylecun/mnist":
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

        elif self.dataset == "uoft-cs/cifar10":
            x = self.block1(x)
            x = self.block2(x)
            x = self.gap(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return F.log_softmax(x, dim=1)

        elif self.dataset == "scikit-learn/adult-census-income":
            if x.dim() > 2:
                x = torch.flatten(x, 1)
            x = x.float()
            x = F.relu(self.tab1(x))
            x = self.tab_drop(x)
            x = F.relu(self.tab2(x))
            x = self.tab_drop(x)
            logits = self.tab_out(x)
            return logits.squeeze(1)


import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

fds = None  # keep your global

num_cols = [
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
]
cat_cols = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
]


def _adult_label_to_float(series):
    mapping = {
        "<=50K": 0.0,
        ">50K": 1.0,
        "<=50K.": 0.0,
        ">50K.": 1.0,
        0: 0.0,
        1: 1.0,
        0.0: 0.0,
        1.0: 1.0,
    }

    def map_one(v):
        if isinstance(v, (int, float)):
            if v in (0, 0.0, 1, 1.0):
                return float(v)
        if isinstance(v, str):
            s = v.strip()
            if s in mapping:
                return mapping[s]
        raise ValueError(f"Unexpected label value for Adult dataset: {v!r}")

    return series.apply(map_one)


class AdultTransform:
    def __init__(self, pre, num_cols, cat_cols, label_col):
        self.pre = pre
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.label_col = label_col

    @staticmethod
    def _label_to_float(series):
        return _adult_label_to_float(series)

    def __call__(self, batch):
        X_df = pd.DataFrame(
            {
                **{c: batch[c] for c in self.num_cols},
                **{c: batch[c] for c in self.cat_cols},
            }
        )
        X_np = self.pre.transform(X_df).astype(np.float32)
        batch["image"] = torch.tensor(X_np, dtype=torch.float32)

        y_series = pd.Series(batch[self.label_col])
        y_np = self._label_to_float(y_series).astype(np.float32).to_numpy()
        batch["label"] = torch.tensor(y_np, dtype=torch.float32)
        return batch


def _vision_apply_transforms(batch):
    data_name = "image" if "image" in batch else "img"
    if data_name == "image":  # mnist
        transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    elif data_name == "img":  # cifar
        transforms = Compose(
            [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
        )
    else:
        raise ValueError(
            "Unknown vision batch format: expected 'image' or 'img' in batch"
        )

    batch["image"] = torch.stack([transforms(img) for img in batch[data_name]])
    if data_name == "img":
        del batch["img"]
    return batch


def load_data(
    context,
    partition_id: int,
    num_partitions: int,
    seed=42,
    dataset="ylecun/mnist",
    dirichlet=False,
    alpha=0.1,
):

    dataset_name = context.run_config["dataset"]
    dataset_name = dataset.replace("/", "-")
    dataset_name = (
        dataset_name + "-iid"
        if not dirichlet
        else dataset_name + "-dirichlet" + f"-{alpha}"
    )

    cache_dir = f"./data/{context.run_config['experiment']}-{dataset_name}-N{context.run_config['num-sampled-clients']}"

    # cache_dir = "./data-supermicro-01"
    os.makedirs(cache_dir, exist_ok=True)
    train_path = os.path.join(
        cache_dir, f"trainloader_{dataset_name}_{partition_id}_of_{num_partitions}.pt"
    )
    test_path = os.path.join(
        cache_dir, f"testloader_{dataset_name}_{partition_id}_of_{num_partitions}.pt"
    )

    # cached?
    if os.path.exists(train_path) and os.path.exists(test_path):
        trainloader = torch.load(train_path, weights_only=False)
        testloader = torch.load(test_path, weights_only=False)

        return trainloader, testloader

    global fds
    if fds is None:
        if dirichlet:
            partition_by_col = (
                "income" if dataset == "scikit-learn/adult-census-income" else "label"
            )
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by=partition_by_col,
                alpha=alpha,
                min_partition_size=10,
                self_balancing=True,
            )
        else:
            partitioner = IidPartitioner(num_partitions=num_partitions)

        fds = FederatedDataset(dataset=dataset, partitioners={"train": partitioner})

    partition = fds.load_partition(partition_id)

    partition_train_test = partition.train_test_split(test_size=0.2, seed=seed)

    if dataset == "scikit-learn/adult-census-income":
        try:
            schema = _load_adult_schema()
        except Exception:
            schema = _compute_and_cache_adult_schema_from_split(
                partition_train_test["train"]
            )

        pre = build_adult_preprocessor_from_schema(
            schema, partition_train_test["train"]
        )
        adult_transform = AdultTransform(
            pre, schema["num_cols"], schema["cat_cols"], schema["label_col"]
        )
        partition_train_test = partition_train_test.with_transform(adult_transform)
    else:
        partition_train_test = partition_train_test.with_transform(
            _vision_apply_transforms
        )

    # Dataloaders
    trainloader = DataLoader(partition_train_test["train"], batch_size=64, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=64)

    # Cache
    torch.save(trainloader, train_path)
    torch.save(testloader, test_path)
    return trainloader, testloader


def hyperparams(mode):
    if mode == "dry":
        lr, lr_decay, momentum = [0.001], [0.9], [0.9]
    else:
        momentum = [0, 0.9]
        lr = [
            0.5,
            0.1,
            0.05,
            0.0005,
            0.0001,
            0.000001,
            0.0000005,
            0.0000001,
            0.00000005,
            0.00000001,
        ]
        lr_decay = [0.0, 0.1, 0.25, 0.99, 1.0]

    hps = [
        {"id": obj_idx, "lr": obj[0], "lr_decay": obj[1], "momentum": obj[2]}
        for obj_idx, obj in enumerate(itertools.product(lr, lr_decay, momentum))
    ]
    # random.shuffle(hps)

    return hps


def hyperparam_by_id(id, mode):
    hps = hyperparams(mode)
    for h in hps:
        if h["id"] == id or id == -1:
            return h


def train(net, trainloader, valloader, epochs, lr, lr_decay, momentum, device):

    print(f"########### Using device: {device}")

    net.to(device)  # move model to GPU if available
    ce_loss = torch.nn.CrossEntropyLoss().to(device)
    bce_loss = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)
    net.train()
    for _ in tqdm(range(epochs)):
        for batch in trainloader:
            if "image" in batch:
                images = batch["image"].to(device)
            else:
                images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()

            outputs = net(images)

            if _is_adult(net):
                loss = bce_loss(outputs.view(-1), labels.view(-1).float())
            else:
                loss = ce_loss(outputs, labels.long())

            loss.backward()
            optimizer.step()
        scheduler.step()

    train_loss, train_acc = test(net, trainloader, device)
    val_loss, val_acc = test(net, valloader, device)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results


def _is_adult(net):
    return getattr(net, "dataset", "") == "scikit-learn/adult-census-income"


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    correct, total_loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"] if "image" in batch else batch["img"]
            labels = batch["label"]

            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)

            if _is_adult(net):
                # outputs: [B] or [B,1]; labels: float {0,1}
                logits = outputs.view(-1)
                y_float = labels.view(-1).float()
                total_loss += bce(logits, y_float).item()

                preds = (torch.sigmoid(logits) >= 0.5).long()
                correct += (preds == y_float.long()).sum().item()
            else:
                # MNIST/CIFAR: outputs [B,C]; labels Long
                y_long = labels.long()
                total_loss += ce(outputs, y_long).item()
                preds = outputs.argmax(dim=1)
                correct += (preds == y_long).sum().item()

    accuracy = correct / len(testloader.dataset)
    return total_loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
