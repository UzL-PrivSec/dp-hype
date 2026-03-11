mode=$1
device=$2
opt=$3

if [ "$mode" != "dry" ] && [ "$mode" != "artifacts" ] && [ "$mode" != "paper" ]; then
  echo "Invalid mode. Must be 'dry', 'artifacts', or 'paper'."
  exit 1
fi

if [ "$device" != "cpu" ] && [[ "$device" != cuda* ]]; then
  echo "Invalid device. Must be 'cpu' or start with 'cuda'."
  exit 1
fi

use_opt="no"
if [ "$opt" == "yes" ]; then
  use_opt="yes"
fi

datasets=("scikit-learn/adult-census-income" "ylecun/mnist" "uoft-cs/cifar10")
ns=(50 100 250)

if [ "$mode" == "dry" ]; then
  datasets=("scikit-learn/adult-census-income")
  ns=(50)
fi

for dataset in "${datasets[@]}"
do
  for num_clients in "${ns[@]}"
  do
    cd /workspace/dp-hype/algorithms/dphype_topk

    if [ "$mode" == "dry" ]; then
      num_runs=1
      local_epochs=1
      eval_rounds=1
    elif [ "$mode" == "artifacts" ]; then
      num_runs=10
    elif [ "$mode" == "paper" ]; then
      num_runs=20
    fi

    if [ "$mode" != "dry" ]; then
        if [ "$dataset" != "uoft-cs/cifar10" ]; then
          local_epochs=5
          eval_rounds=5
        else
          local_epochs=10
          eval_rounds=10
        fi
    fi

    flwr run . privutil_tradeoff_n${num_clients} --run-config "dataset=\"$dataset\" num-sampled-clients=$num_clients num_runs=$num_runs local_epochs=$local_epochs eval_rounds=$eval_rounds non_iid=\"no\" alpha=0.0 baseline_opt=\"$use_opt\" mode=\"$mode\" device=\"$device\" resource_monitoring=\"no\""

    cd /workspace/dp-hype/plotting

    julia plot_privutil_tradeoff.jl -d $dataset -n $num_clients -a 0.0 -m $mode
  done
done