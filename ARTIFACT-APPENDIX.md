# Artifact Appendix (Required for all badges)

Paper title: **DP-Hype: Federated Differentially Private Hyperparameter Search**

Requested Badge(s):
  - [x] **Available**
  - [x] **Functional**
  - [x] **Reproduced**

## Description
Title: "DP-Hype: Federated Differentially Private Hyperparameter Search"

Authors:
   - Johannes Liebenow, University of Lübeck
   - Thorsten Peinemann, University of Lübeck
   - Esfandiar Mohammadi, University of Lübeck

Description: This repository provides code to automate, reproduce and plot **all** the experiments done in the above-mentioned paper. The majority of experiments require access to an NVIDIA GPU and enough CPU power. As reproducing all experiments requires multiple days in a row, the artifacts contain data for the already existing baselines to reduce the runtime and ease comparison. Code to reproduce the baselines is still available, though.  

### Security/Privacy Issues and Ethical Concerns
No security / privacy / ethical issues or concerns.

## Basic Requirements

### Hardware Requirements

1. Minimal Hardware Requirements: 
A multi-core CPU setup with around 10GB RAM. A GPU is also required as it greatly speeds up the training process. We simulate each client in our federated learning setup as a separate process using the Flower framework. Each client requires a single core, around 1GB of RAM and 200MB of vRAM for the largest data set Cifar-10. The resource allocation per client can be configured in the file "config.toml". Note that fractional CPUs and GPUs are possible but overcommitment should be avoided as clients will be too slow or fail in the worst case.

2. Hardware used for Paper Experiments:
We performed our experiments on eight cores of an AMD EPYC 7513 CPU and on an A100-20C GPU in MIG setup with 20GB of VRAM. In the "config.toml" we set "num-cpus" = 1 and "num-gpus"=0.12 (~= 1 / #available_cores) to run a total of eight clients in parallel.

**TODO: Please adjust the "config.toml" file according to your hardware setup (number of Cores and GPUs) in order to run the maximum number of clients in parallel.**

### Software Requirements

Only the repository has to be cloned. All dependencies, requirements, data sets and models are managed automatically by the Dockerfile.

1. OS: Any OS with AMD64 (x86-64) architecture (tested on Ubuntu 22.04)

2. OS Packages: Docker has to be installed and the GPUs have to be configured properly so that they can either be accessed from inside a Docker container directly or forwarded via a pre-defined runtime.

3. Docker Setup:
   - Docker Version: Docker 24.0.7 in Swarm mode
   - Docker Image: pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime. The CUDA and cuDNN versions have to be compatible with the GPU setup of the OS.

**TODO: Please choose an image from [PyTorch Docker Hub](https://hub.docker.com/r/pytorch/pytorch/tags) which is compatible with your CUDA version and adjust the first line in the Dockerfile accordingly.** 

4. Programming Languages: Python 3.11.9 and Julia 1.12.1

5. Artifact Packages:
   - Python Requirements: "requirements.txt" and PyTorch 2.4.0 preinstalled 
   - Julia Requirements: "Project.toml"
   - Installed packages: git wget tar texlive 

6. Machine Learning Models: Only small (C)NNs: see "class Net" in "algorithms/dphype_topk/dphype/task.py".

7. Data Sets: [UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult), [MNIST Database](https://en.wikipedia.org/wiki/MNIST_database), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).
   All data sets are publicly available, e.g., via flower_datasets.

### Estimated Time and Storage Consumption

- Human and Compute Time: Running all our experiments sequentially requires over a week on our hardware setup. To reduce it to only a few days, our artifacts contain a mode called "artifacts", which does not perform the full number of runs but already suffices to produce similar figures.

- Overall Disk Space: Only the Docker image (12GB) and the three data sets (300MB) take up a noteworthy portion of disk space. Our experiments only produce dataloader partitions, CSV and PDF files, with the latter requiring only a few MBs.  

## Environment

### Accessibility

[GitHub Repository](https://github.com/UzL-PrivSec/dp-hype/tree/main)  

### Set up the environment

Clone the GitHub repository of the artifacts:
```bash
git clone https://github.com/UzL-PrivSec/dp-hype.git
```

Create the corresponding Docker image and container:
```bash
cd dp-hype
docker build -t dp-hype-artifacts .
```
The building process of the image may take a while, because the base image has to be downloaded and all dependencies and packages are being installed. If the image builds successfully, then you can start creating a container:

```bash
docker run -it --gpus all --name dp-hype-artifacts dp-hype-artifacts:latest bash 
```
After the building process has finished successfully, you should be automatically attached to a bash inside the running container. You can detach without exiting the shell using the escape sequence ```Ctrl+P Ctrl+Q``` and attach again via ```bash docker container attach dp-hype-artifacts```. Commands can either be executed from within the Docker container or from the host using `docker container exec -d dp-hype-artifacts <command>`.

Depending on your GPU setup and your Docker version, there are two possible ways of accessing GPUs from inside a container. You either have a pre-configured runtime "nvidia", which can be passed to the container via "--runtime=nvidia", or you have installed the nvidia-container-toolkit, which allows you to pass all (--gpus all) or specific devices (--gpus device=0).

### Testing the Environment

The artifacts provide a script "test_setup.sh", which can be used to check if the main components (Python, Julia, Dependencies, GPU / CUDA) are correctly set up. It also performs a few test runs of the simulation and DP-Hype. You can execute it using
```bash
source test_setup.sh
``` 
Once executed, it lists each check separately along with a corresponding success or error message. If all checks are passed successfully, you can continue with the artifact evaluation.

## Artifact Evaluation

### Main Results and Claims

#### Main Result 1: Simulations (Figure 2)

This figure contains three subfigures where we simulate the probability of DP-Hype selecting good hyperparameters. For that, we omit the local training part and create a set of local losses by drawing from Gaussian distributions, one for good and one for bad hyperparameters. The experiments were performed for two privacy budgets, 0.25 and 1.0, for 250 clients and multiple ks (1, 5, 25, 50, 100).
- In Figure 2a, we vary the standard deviation of both Gaussian distributions so that higher sigmas result in more overlap between losses of good and bad hyperparameters. For the small privacy budget, the influence of DP noise becomes visible and interestingly, not the smallest k (the one with the smallest noise), but k=5 performs best as it has a higher chance that one of the best hyperparameter candidates receives the most votes. For a privacy budget of 1.0, the noise is small enough such that all ks perform well, as without much noise, the best hyperparameter candidate will almost always receive the most votes.
- In Figure 2b, sigma is fixed and the percentage of good hyperparameters varies. For the smaller privacy budget, we can see that every k has its own sweet spot for when it fits best. The most important observation is that smaller ks perform well for a smaller number of good hyperparameters and vice versa. For larger privacy budgets this effect disappears but k=100 is still bad, as it is essentially random selection. The probability of selecting a good candidate merely increases with the number of good candidates.
- In Figure 2c, we vary the number of total hyperparameters while leaving sigma and the number of good hyperparameters fixed. For small privacy budgets, small ks perform better because they naturally reduce the number of votes for bad candidates. This effect degrades because with a lot of hyperparameters the probability of large noise changing the voting increases. A privacy budget of 1.0 again has less noise, which results in a good performance of nearly all ks. k=100 is an exception again, because having to vote for 100 candidates when there are 100 in total is basically random guessing.  

#### Main Result 2: Privacy-Utility Trade-Off of DP-Hype (Figure 4, 5, 12, 13)

All figures show the privacy-utility trade-off in terms of accuracy of DP-Hype in comparison to the best possible choice (Opt) and random guessing (RandGuess). In every plot, the x-axis shows different privacy budgets (0.1, 0.25, 0.5, 1.0, 3.0, inf) and the y-axis always shows the accuracy when training a federated model on the hyperparameter chosen by DP-Hype. Figure 4 shows results for the iid setting and Figures 5, 12, and 13 cover the non-iid settings for 50, 100, and 250 clients, respectively. All experiments were performed on the three data sets. The main claim of these figures can be summarized as follows: DP-Hype stays close to the non-private baseline (OPT), however at some point the privacy-utility trade-off kicks in and lowers its accuracy. These cut-off points heavily depend on the number of clients and the non-iid degree, and are unique for each data set. 

#### Main Result 3: Accuracy Distributions (Figure 3, 8)

Both figures show histograms over the accuracy values per hyperparameter for all data sets for a specific scenario. Figure 3 shows such a distribution for evaluating each hyperparameter globally and Figure 8 shows the same distribution for various numbers of clients and non-iid settings. 

#### Main Result 4: Resource Consumption (Table 4)

This table captures the resource consumption of DP-Hype from a client's perspective and can be divided into two parts, the local hyperparameter tuning and the secure summation protocol.
- For the first part, we analyze the maximum vRAM requirement as well as the runtime for local hyperparameter tuning for all numbers of clients and all data sets. The vRAM requirement only depends on the data sets and not on the number of clients, because the batch size always stays the same independent of the number of data points per client. Adult requires the least vRAM, followed by MNIST and Cifar-10 the most, as it consists of large images with three color channels. The runtime depends on both the number of clients and the data set; the order of required runtime corresponds to that of vRAM.
- The federated voting part captures the bandwidth (send and received bits per client) as well as the runtime for a single client. Secure summation only depends on the number of clients, as the voting vector is independent of the actual data set. More clients require more bandwidth and runtime because the cryptographic mechanisms behind secure summation such as secret sharing scale with the number of clients.

#### Main Result 5: Dirichlet Distributions (Figure 6, 7)

Figures 6 and 7 show what the data / label distributions look like for 10 clients in various non-iid settings.  

#### Main Result 6: Ablation Raw Losses (Figure 9)

This figure shows a comparison in terms of the privacy-utility trade-off between DP-Hype and a slight modification of it, called RawLosses, where the accuracy is sent directly instead of zeros and ones. The experiments were performed for 100 clients on the Cifar-10 data set for beta=5.0 and beta=0.5. This figure shows that DP-Hype outperforms the alternative strategy. 

#### Main Result 7: Ablation Subsets (Figure 10)

This figure shows a comparison in terms of the privacy-utility trade-off between DP-Hype and a simple privacy-preserving parallelization strategy we call Subsets. Instead of letting each client evaluate all the hyperparameters locally, clients are partitioned into subsets and a single hyperparameter is assigned to each subset. The experiments were performed on MNIST and Cifar-10 for 250 (beta=30) and 100 clients (iid), respectively. This figure is used to show that the computational overhead imposed by DP-Hype provides a massive utility boost in contrast to parallelizing the local hyperparameter evaluation. 

#### Main Result 8: Ablation Feathers (Figure 11)

This figure shows a comparison in terms of the privacy-utility trade-off between DP-Hype and an algorithm from prior work called Feathers. The experiments were performed for 250 clients, in the iid setting for Cifar-10 and MNIST. This ablation is used to show that DP-Hype outperforms Feathers in most cases even in iid scenarios. 

#### Main Result 9: DP Noise Scale (Section 6.3 First Paragraph)

The standard deviations of the distribution used to draw noise to achieve DP can be obtained by using the script "dp_noise_scale.py". 

### Experiments

The experiments are arranged in such a way that allows reproducing all the different types of results while reducing the runtime as much as possible. Executing experiments 1–9 reproduces all figures from the paper except for Figure 5 and 12. This is because the runtimes of experiments 10 and 11 cannot be reduced further, as the non-iid settings in combination with the noise of differential privacy require a lot of runs to obtain reliable averages. However, experiments 3 and 4 are very similar to experiments 10 and 11, but the latter two have a much larger runtime. 

Every experiment creates CSV files, and PDF figures in `figures/<mode>/`. These figures are equivalent to those in the paper, though more granularly separated per experiment. Each experiment also creates its corresponding legend. To check if an experiment has finished, verify that the corresponding plot exists using `docker container exec dp-hype-artifacts ls /workspace/dp-hype/figures/<mode>/`. Figures can be copied to the host system using `docker container cp dp-hype-artifacts:/workspace/dp-hype/figures/<mode>/<figure_name>.pdf <destination>`.

Every experiment supports multiple modes (`<mode>`):
- `dry`: Used inside the script "test_setup.sh" to test the entire pipeline of experiments and plotting. This mode does **not** produce any meaningful results.
- `paper`: Performs the exact same experiments as in the paper. The runtime estimations below were derived from the artifacts mode. Reproducing the exact same experiments requires roughly two weeks on the tested setup without parallelization.  
- `artifacts`: The intended mode to evaluate these artifacts, with two major differences from `paper`. First, in experiment 3 and 4, DP-Hype lets each client perform local hyperparameter evaluations only once, but performs multiple runs to capture the influence of differential privacy. Second, the number of runs is reduced by half for all experiments. This mode reproduces very similar figures compared to those in the paper while significantly reducing the runtime for most experiments. We strongly encourage running experiments in parallel if possible.  

To reduce runtime further, we include the result files for the OPT baseline. OPT is an often evaluated, standard federated learning algorithm. Plotting scripts in "plotting/" will use the baseline results from "opt_data/", indicated by a warning, if the corresponding data was not reproduced yet. Still, these artifacts provide code to reproduce the baseline results by appending a "yes" to the commands of experiments 3, 4, 10, and 11.

**TODO: Replace `<device>`, e.g. `cuda:0`,  with the corresponding GPU device name or `cpu` for testing.**

*Optional: Use `screen` or `tmux` within the container to start a session that can easily be detached.*

#### Experiment 1: Simulations
- Corresponding Result: Main Result 1
- Time: ~0.5 hours
- Command: 
```bash
bash experiments/simulations.sh artifacts
```

Fully contains the experiment described in Main Result 1.  The CSV file is stored in algorithms/simulation/results/artifacts/ and the resulting figures are "figure2a_simulation_sigma.pdf", "figure2b_simulation_hgood.pdf" and "figure2c_simulation_htotal.pdf". 

#### Experiment 2: Resource Consumption
- Corresponding Result: Main Result 4
- Time: ~0.5 hours
- Command: 
```bash
bash experiments/resource_consumption.sh artifacts <device>  
```

Fully contains the experiment described in Main Result 4.  The CSV files are stored in algorithms/secsum_monitor/results/artifacts/ and algorithms/dphype_topk/dphype/results/artifacts. The resulting table is "table4_resource_consumption_table.pdf". 

#### Experiment 3: Privacy-Utility Trade-Off IID (Figure 4)
- Corresponding Result: Main Result 2
- Time: ~12 hours
- Command: 
```bash
bash experiments/privutility_tradeoffs_iid.sh artifacts <device>   
```

Starts experiments for the privacy-utility trade-off of DP-Hype for all client counts (50, 100, 250) on all data sets in the iid setting.
9 CSV files with the results will be created and stored in "algorithms/dphype_topk/dphype/results/artifacts/".
The same number of figures will be created. The names are figure4_privutil_tradeoff_<Adult|MNIST|Cifar-10>_N<50|100|250>_iid.pdf.

#### Experiment 4: Privacy-Utility Trade-Off NON-IID N=50 (Figure 13)
- Corresponding Result: Main Result 2
- Time: ~12 hours
- Command: 
```bash
bash experiments/privutility_tradeoffs_noniid.sh artifacts <device> 50   
```

Starts experiments for the privacy-utility trade-off of DP-Hype for 50 clients on all data sets in three non-iid scenarios.
9 CSV files with the results will be created and stored in "algorithms/dphype_topk/dphype/results/artifacts/".
The same number of figures will be created. The names are figure13_privutil_tradeoff_<Adult|MNIST|Cifar-10>_N50_dirichlet-<30.0|5.0|0.5>.pdf.

#### Experiment 5: Ablation Subsets 
- Corresponding Result: Main Result 7
- Time: ~1 hour
- Command: 
```bash
bash experiments/ablation_subsets.sh artifacts <device>    
```

Fully contains the experiment described in Main Result 7. The CSV file is stored in algorithms/dphype_subsets/dphype/results/artifacts/ and the resulting figure has the name "figure10_ablation_subsets.pdf".

*Hint: Relies on data from experiments 3, 11 but can be executed without this data to only reproduce Subsets.*

#### Experiment 6: Ablation Raw Losses
- Corresponding Result: Main Result 6
- Time: ~2.5 hours
- Command: 
```bash
bash experiments/ablation_rawlosses.sh artifacts <device>  
```

Fully contains the experiment described in Main Result 6. The CSV file is stored in algorithms/dphype_subsets/dphype/results/artifacts/ and the resulting figure has the name "figure9_ablation_rawlosses.pdf". 

*Hint: Relies on data from experiment 10 but can be executed without this data to only reproduce RawLosses.*

#### Experiment 7: Accuracy Distributions
- Corresponding Result: Main Result 3
- Time: Instant
- Command: 
```bash
bash experiments/acc_distributions.sh artifacts  
```

Fully contains the experiment described in Main Result 3. The resulting figures are called "figure3_acc_distribution_iid.pdf", "figure8a_acc_distribution_noniid_a30allN.pdf", "figure8b_acc_distribution_noniid_a5allN.pdf" and "figure8c_acc_distribution_noniid_a05allN.pdf". 

#### Experiment 8: Dirichlet Distributions
- Corresponding Result: Main Result 5
- Time: Instant
- Command: 
```bash
bash experiments/dirichlet_distributions.sh artifacts
```

Fully contains the experiment described in Main Result 5. The resulting figures are called "figure6_dirichlet_distribution.pdf" and "figure7_dirichlet_distribution.pdf". 

#### Experiment 9: Ablation Feathers 
- Corresponding Result: Main Result 8
- Time: ~30 hours
- Command: 
```bash
bash experiments/ablation_feathers.sh artifacts <device>   
```

Fully contains the experiment described in Main Result 8. The CSV file is stored in algorithms/dphype_feathers/dphype/results/artifacts/ and the resulting figure has the name "figure11_ablation_feathers.pdf". 

*Hint: Relies on data from experiment 3 but can be executed without this data to only reproduce the results of Feathers.*

#### Experiment 10: Privacy-Utility Trade-Off NON-IID N=100 (Figure 5)
- Corresponding Result: Main Result 2
- Time: multiple days
- Command: 
```bash
bash experiments/privutility_tradeoffs_noniid.sh artifacts <device> 100   
```

Starts experiments for the privacy-utility trade-off of DP-Hype for 100 clients on all data sets in three non-iid scenarios.
9 CSV files with the results will be created and stored in "algorithms/dphype_topk/dphype/results/artifacts/".
The same number of figures will be created. The names are figure5_privutil_tradeoff_<Adult|MNIST|Cifar-10>_N100_dirichlet-<30.0|5.0|0.5>.pdf.

#### Experiment 11: Privacy-Utility Trade-Off NON-IID N=250 (Figure 12)
- Corresponding Result: Main Result 2
- Time: multiple days
- Command: 
```bash
bash experiments/privutility_tradeoffs_noniid.sh artifacts <device> 250
```

Starts experiments for the privacy-utility trade-off of DP-Hype for 250 clients on all data sets and in iid as well as three non-iid scenarios.
9 CSV files with the results will be created and stored in "algorithms/dphype_topk/dphype/results/artifacts/".
The same number of figures will be created. The names are figure12_privutil_tradeoff_<Adult|MNIST|Cifar-10>_N250_dirichlet-<30.0|5.0|0.5>.pdf.

## Limitations

- A Note on Differential Privacy: DP-Hype is a randomized algorithm in order to satisfy differential privacy. This additional randomness requires a large number of iterations to reproduce the results in the paper without large deviations. However, the required number of iterations can be very large, especially for privacy budgets <= 0.5. Thus, one should focus on the results for higher privacy budgets first, because they can be reproduced more reliably. The tendency of smaller privacy budgets to reduce performance drastically should also be visible, yet exact averages may be hard to reproduce within a limited time frame. This is especially true for results with a high confidence interval depicted as transparent, so-called ribbons in the figures.       
- The runtime in the resource consumption table (Table 4) will likely deviate when the artifacts are evaluated on different hardware. However, this is only natural and does not invalidate the comparison between runtimes.
- The random guessing baseline (RandGuess) in Figure 4,5,12,13 could slightly deviate as performing the OPT baseline experiments still has some variance.
- The accuracy histograms (Figure 3,8) may deviate a bit when reproduced. However, deviating counts from one accuracy bucket will stay in the neighborhood of that bucket and will likely not move to distant buckets. In other words, the height of some bars will likely change, yet the overall statement will not.

## Notes on Reusability

- Our main algorithm DP-Hype is embedded into the federated machine learning framework Flower as well as most of the other evaluated algorithms. Thus, other settings such as the number of users, non-iid scenarios and other data sets can easily be realized.
