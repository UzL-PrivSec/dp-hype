FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV GKSwstype=nul \
    JULIA_PKG_PRECOMPILE_AUTO=0 \
    JULIA_PROJECT=/workspace/dp-hype \
    FLWR_HOME=/workspace/dp-hype/

WORKDIR /workspace

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    nano git wget tar screen tmux htop ca-certificates texlive-latex-base texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

# Make screen use bash instead of sh
RUN printf '%s\n' \
  'shell /bin/bash' \
  'defshell -bash' \
  > /etc/screenrc

# Install Julia
RUN wget --no-verbose https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-1.12.1-linux-x86_64.tar.gz \
    && tar -xzf julia-1.12.1-linux-x86_64.tar.gz \
    && rm julia-1.12.1-linux-x86_64.tar.gz

ENV PATH="/workspace/julia-1.12.1/bin:${PATH}"

# Copy repo
RUN mkdir dp-hype 
COPY . ./dp-hype

WORKDIR /workspace/dp-hype

# Python deps
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# Julia deps
RUN julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Create result directories
RUN bash -lc 'mkdir -p \
    .temp \
    algorithms/dphype_topk/dphype/results/{dry,artifacts,paper} \
    algorithms/dphype_rawlosses/dphype/results/{dry,artifacts,paper} \
    algorithms/dphype_subsets/dphype/results/{dry,artifacts,paper} \
    algorithms/simulation/results/{dry,artifacts,paper} \
    algorithms/secsum_monitor/results/{dry,artifacts,paper} \
    algorithms/feathers/fl_dp_sa/results/{dry,artifacts,paper} \
    figures/{dry,artifacts,paper}'
