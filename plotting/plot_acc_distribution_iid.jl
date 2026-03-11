using StatsPlots
using CSV
using DataFrames
using Measures
using LaTeXStrings
using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "--mode", "-m"
    help = "Mode: dry, artifacts, paper"
    arg_type = String
    required = true
end
parsed_args = parse_args(s)
mode = parsed_args["mode"]
if mode ∉ ["dry", "artifacts", "paper"]
    @error "Invalid -m value '$mode'. Must be one of: dry, artifacts, paper"
    exit(1)
end

try
    global df_mnist = CSV.read("/workspace/dp-hype/algorithms/dphype_topk/dphype/results/$mode/opt-ylecun-mnist-iid-N50-epsAll.csv", DataFrame)
    global df_cifar = CSV.read("/workspace/dp-hype/algorithms/dphype_topk/dphype/results/$mode/opt-uoft-cs-cifar10-iid-N50-epsAll.csv", DataFrame)
    global df_adult = CSV.read("/workspace/dp-hype/algorithms/dphype_topk/dphype/results/$mode/opt-scikit-learn-adult-census-income-iid-N50-epsAll.csv", DataFrame)
catch e
    @warn "Opt result file not found -> Falling back to /workspace/dp-hype/opt_data/. Or execute 'experiments/privutility_tradeoff.sh $mode <device> 100 yes' first" exception = e
    try
        global df_mnist = CSV.read("/workspace/dp-hype/opt_data/opt-ylecun-mnist-iid-N50-epsAll.csv", DataFrame)
        global df_cifar = CSV.read("/workspace/dp-hype/opt_data/opt-uoft-cs-cifar10-iid-N50-epsAll.csv", DataFrame)
        global df_adult = CSV.read("/workspace/dp-hype/opt_data/opt-scikit-learn-adult-census-income-iid-N50-epsAll.csv", DataFrame)
    catch e
        @error "Ooops something went wrong." exception = e
        exit(1)
    end
end

default(
    guidefont=font(12),
    tickfont=font(12),
    legendfont=font(12),
    titlefont=font(12),
    lw=1,
    markersize=6,
    markerstrokewidth=10,
)

lay = @layout([a; b{0.15h}])

p = plot(
    layout=lay,
    size=(600, 300),
    left_margin=5mm, right_margin=5mm, top_margin=5mm,
    bottom_margin=2mm,            # small overall bottom margin
    legend=false,             # no per-subplot legends
)

@df df_mnist histogram!(p, subplot=1, :accuracy, bins=range(0.0, 1.0, 50), label="MNIST", alpha=0.5)
@df df_cifar histogram!(p, subplot=1, :accuracy, bins=range(0.0, 1.0, 50), label="CIFAR-10", alpha=0.5)
@df df_adult histogram!(p, subplot=1, :accuracy, bins=range(0.0, 1.0, 50), label="Adult", alpha=0.5)
xlabel!(p, subplot=1, "Accuracy")
ylabel!(p, subplot=1, "Frequency")

for (dt_name, linestyle) in zip(["MNIST", "CIFAR-10", "Adult"], [:solid, :dot, :dashdotdot])
    plot!(p, [NaN], [NaN];
        subplot=2,
        label=dt_name)
end

plot!(p; subplot=2,
    legend=:bottom,
    legend_column=-1,
    legend_title="Data Sets",
    xaxis=false, yaxis=false, grid=false, framestyle=:none)

savefig("/workspace/dp-hype/figures/$mode/figure3_acc_distribution_iid.pdf")

