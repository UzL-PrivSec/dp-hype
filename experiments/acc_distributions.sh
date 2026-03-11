mode=$1

if [ "$mode" != "dry" ] && [ "$mode" != "artifacts" ] && [ "$mode" != "paper" ]; then
  echo "Invalid mode. Must be 'dry', 'artifacts', or 'paper'."
  exit 1
fi

cd /workspace/dp-hype/plotting

julia plot_acc_distribution_iid.jl -m $mode
    
julia plot_acc_distribution_noniid.jl -m $mode