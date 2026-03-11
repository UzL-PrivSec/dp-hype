mode=$1

if [ "$mode" != "dry" ] && [ "$mode" != "artifacts" ] && [ "$mode" != "paper" ]; then
  echo "Invalid mode. Must be 'dry', 'artifacts', or 'paper'."
  exit 1
fi

cd /workspace/dp-hype/plotting

python3 plot_dirichlet_distribution.py --mode $mode