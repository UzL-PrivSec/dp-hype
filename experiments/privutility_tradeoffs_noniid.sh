mode=$1
device=$2
num_clients=$3
opt=$4

if [ "$mode" != "dry" ] && [ "$mode" != "artifacts" ] && [ "$mode" != "paper" ]; then
  echo "Invalid mode. Must be 'dry', 'artifacts', or 'paper'."
  exit 1
fi

if [ "$device" != "cpu" ] && [[ "$device" != cuda* ]]; then
  echo "Invalid device. Must be 'cpu' or start with 'cuda'."
  exit 1
fi

if [ "$num_clients" != "50" ] && [ "$num_clients" != "100" ] && [ "$num_clients" != "250" ]; then
  echo "Invalid num_clients. Must be 50, 100, or 250."
  exit 1
fi

use_opt="no"
if [ "$opt" == "yes" ]; then
  use_opt="yes"
fi

datasets=("scikit-learn/adult-census-income" "ylecun/mnist" "uoft-cs/cifar10")
alphas=("30.0" "5.0" "0.5")

if [ "$mode" == "dry" ]; then
  if [ $num_clients -eq 100 ]; then
    datasets=("ylecun/mnist")
    alphas=("30.0")
  elif [ $num_clients -eq 250 ]; then
    datasets=("uoft-cs/cifar10")
    alphas=("5.0")
  fi
fi

for dataset in "${datasets[@]}"
do
  for alpha in "${alphas[@]}"
  do
    cd /workspace/dp-hype/algorithms/dphype_topk

    if [ "$num_clients" -eq 250 ] && [ "$dataset" == "scikit-learn/adult-census-income" ] && [ "$alpha" == "0.5" ]; then
      alpha=2.0
    fi

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
      else
        local_epochs=10
      fi
      eval_rounds=5
    fi

    flwr run . privutil_tradeoff_n${num_clients} --run-config "dataset=\"$dataset\" num-sampled-clients=$num_clients num_runs=$num_runs local_epochs=$local_epochs eval_rounds=$eval_rounds non_iid=\"yes\" alpha=$alpha baseline_opt=\"$use_opt\" mode=\"$mode\" device=\"$device\" resource_monitoring=\"no\""

    cd /workspace/dp-hype/plotting

    julia plot_privutil_tradeoff.jl -d $dataset -n $num_clients --non_iid -a $alpha -m $mode
  done
done