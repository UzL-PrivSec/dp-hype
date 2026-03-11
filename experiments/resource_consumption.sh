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

cd /workspace/dp-hype/algorithms/secsum_monitor

python3 main.py --mode $mode

cd /workspace/dp-hype/algorithms/dphype_topk/

datasets=("scikit-learn/adult-census-income" "ylecun/mnist" "uoft-cs/cifar10")
num_clients=(50 100 250)

for dataset in "${datasets[@]}"
do
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

    for num_client in "${num_clients[@]}"
    do
        flwr run . privutil_tradeoff_n${num_client} --run-config "dataset=\"$dataset\" num-sampled-clients=$num_client num_runs=$num_runs local_epochs=$local_epochs eval_rounds=$eval_rounds non_iid=\"no\" alpha=0.0 baseline_opt=\"no\" mode=\"$mode\" device=\"$device\" resource_monitoring=\"yes\""
    done
done

cd /workspace/dp-hype/plotting

julia create_resource_consumption_table.jl -m $mode

cp /workspace/dp-hype/.temp/table4_resource_consumption_table.pdf /workspace/dp-hype/figures/$mode/table4_resource_consumption_table.pdf