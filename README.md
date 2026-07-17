# DP-Hype: Federated Differentially Private Hyperparameter Search

Artifact repository for the PoPETS 2026 paper:

> **DP-Hype: Federated Differentially Private Hyperparameter Search**  
> Johannes Liebenow, Thorsten Peinemann, Esfandiar Mohammadi  
> University of Lübeck

DP-Hype is a federated hyperparameter search algorithm that satisfies differential privacy (DP). Each client locally evaluates a set of hyperparameter candidates and casts a private vote using the top-k mechanism. Votes are securely aggregated via a secure summation protocol, and the hyperparameter with the most votes is selected. 

The artifacts provide fully automated scripts to reproduce and plot all experiments from the paper using Docker. 

## Repository Structure

```
├── algorithms/            # All evaluated algorithms (DP-Hype, baselines, ablations)
├── experiments/           # Shell scripts to run each experiment
├── plotting/              # Julia/Python plotting scripts
├── opt_data/              # Pre-computed OPT baseline results
├── Dockerfile             # Builds the full runtime environment
├── config.toml            # Flower client resource configuration (CPUs/GPUs per client)
├── requirements.txt       # Python dependencies
├── Project.toml           # Julia dependencies
├── test_setup.sh          # Environment validation and testing
├── dp_noise_scale.py      # Utility to compute DP noise scale values
└── ARTIFACT-APPENDIX.md   # Full artifact appendix
```

## Getting Started

The entire environment, including all dependencies, data sets, and models, is managed automatically via Docker, so no manual installation is required. However, a working NVIDIA GPU setup with Docker GPU access (`--gpus all` or `--runtime=nvidia`) is a prerequisite.

All setup instructions, hardware and software requirements, experiment commands, and expected runtimes are documented in detail in [ARTIFACT-APPENDIX.md](ARTIFACT-APPENDIX.md). The appendix also contains a few **TODO** items that require manual adjustment before running the experiments, in particular selecting the correct Docker base image for your CUDA version and configuring the per-client resource allocation in `config.toml`.

## Citation

If you use **DP-Hype** in your work, please cite:

> Johannes Liebenow, Thorsten Peinemann, and Esfandiar Mohammadi.  
> **DP-Hype: Federated Differentially Private Hyperparameter Search.**  
> *Proceedings on Privacy Enhancing Technologies*, 2026.

```bibtex
@article{liebenow2026dp,
  title   = {DP-Hype: Federated Differentially Private Hyperparameter Search},
  author  = {Liebenow, Johannes and Peinemann, Thorsten and Mohammadi, Esfandiar},
  journal = {Proceedings on Privacy Enhancing Technologies},
  year    = {2026}
}
