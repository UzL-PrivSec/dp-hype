datasets=("uoft-cs/cifar10" "ylecun/mnist")
n=(250 100)
non_iids=("yes" "no")
alphas=("30.0" "0.0")

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

for i in "${!non_iids[@]}"
do
  cd /workspace/dp-hype/algorithms/dphype_subsets

  dataset="${datasets[$i]}"
  n="${n[$i]}"
  non_iid="${non_iids[$i]}"
  alpha="${alphas[$i]}"

  if [ "$mode" == "dry" ]; then
    num_runs=1
    local_epochs=1
    eval_rounds=1
  elif [ "$mode" == "artifacts" ]; then
    num_runs=5
  elif [ "$mode" == "paper" ]; then
    num_runs=20
  fi

  if [ "$mode" != "dry" ]; then
    if [ "$non_iid" != "yes" ]; then
      if [ "$dataset" != "uoft-cs/cifar10" ]; then
        local_epochs=5
        eval_rounds=5
      else
        local_epochs=10
        eval_rounds=10
      fi
    else
      if [ "$dataset" != "uoft-cs/cifar10" ]; then
        local_epochs=5
      else
        local_epochs=10
      fi
      eval_rounds=5
    fi
  fi

  flwr run . privutil_tradeoff_n${n} --run-config "dataset=\"$dataset\" num-sampled-clients=$n num_runs=$num_runs local_epochs=$local_epochs eval_rounds=$eval_rounds non_iid=\"$non_iid\" alpha=$alpha mode=\"$mode\" device=\"$device\""
done

cd /workspace/dp-hype/plotting

julia plot_ablation_subsets.jl -m $mode