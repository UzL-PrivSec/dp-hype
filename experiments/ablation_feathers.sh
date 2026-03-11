datasets=("uoft-cs/cifar10" "ylecun/mnist")
eps=("0.1" "0.25" "0.5" "1.0" "3.0" "inf")

mode=$1
device=$2

if [ "$mode" != "dry" ] && [ "$mode" != "artifacts" ] && [ "$mode" != "paper" ]; then
  echo "Invalid mode. Must be 'dry', 'artifacts', or 'paper'."
  exit 1
fi

if [ "$device" != "cpu" ] && [[ "$device" != cuda* ]]; then
  echo "Invalid device. Must be 'cpu' or start with 'cuda'."
  exit 1
fi

cd /workspace/dp-hype/algorithms/feathers

for i in "${!datasets[@]}"
do
  
  dataset="${datasets[$i]}"

  for e in "${eps[@]}"
  do
    if [ "$mode" == "dry" ]; then
      num_runs=1
      local_epochs=1
      rounds=1
      max_exploration_steps=1
    elif [ "$mode" == "artifacts" ]; then
      num_runs=5
      local_epochs=5
      rounds=120
      max_exploration_steps=100
    elif [ "$mode" == "paper" ]; then
      num_runs=10
      local_epochs=5
      rounds=120
      max_exploration_steps=100
    fi

    flwr run . privutil_tradeoff_n250 --run-config "dataset=\"$dataset\" num_runs=$num_runs local_epochs=$local_epochs rounds=$rounds max_exploration_steps=$max_exploration_steps epsilon=\"$e\" mode=\"$mode\" device=\"$device\""
  done
done

cd /workspace/dp-hype/plotting

julia plot_ablation_feathers.jl -m $mode