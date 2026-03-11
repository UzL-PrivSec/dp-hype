mode=$1

if [ "$mode" != "dry" ] && [ "$mode" != "artifacts" ] && [ "$mode" != "paper" ]; then
  echo "Invalid mode. Must be 'dry', 'artifacts', or 'paper'."
  exit 1
fi

cd /workspace/dp-hype/algorithms/simulation

julia simulation_hgood.jl -m $mode

cd /workspace/dp-hype/plotting

julia plot_simulation_hgood.jl -m $mode

# ----

cd /workspace/dp-hype/algorithms/simulation

julia simulation_htotal.jl -m $mode 

cd /workspace/dp-hype/plotting

julia plot_simulation_htotal.jl -m $mode

# ----

cd /workspace/dp-hype/algorithms/simulation

julia simulation_sigma.jl -m $mode 

cd /workspace/dp-hype/plotting

julia plot_simulation_sigma.jl -m $mode