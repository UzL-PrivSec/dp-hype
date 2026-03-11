using StatsPlots
using CSV
using DataFrames
using Measures
using LaTeXStrings
using Statistics
using Distributions
using Random
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


save_dir = "/workspace/dp-hype/figures/$mode"

_feathers_eps_strs = ["0.1", "0.25", "0.5", "1.0", "3.0", "inf"]
_feathers_eps_vals = [0.1, 0.25, 0.5, 1.0, 3.0, Inf]

function load_feathers(base_name, mode)
    dfs = DataFrame[]
    for (eps_str, eps_val) in zip(_feathers_eps_strs, _feathers_eps_vals)
        path = "/workspace/dp-hype/algorithms/feathers/fl_dp_sa/results/$mode/$(base_name)-eps$(eps_str).csv"
        df = CSV.read(path, DataFrame)
        df[!, :eps] .= eps_val
        push!(dfs, df)
    end
    return vcat(dfs...)
end

try
    global df_feather_mnist_n250 = load_feathers("feathers-ylecun-mnist-N250", mode)
    global df_feathers_cifar10_n250 = load_feathers("feathers-uoft-cs-cifar10-N250", mode)
catch e
    @error "Result file not found. Please execute 'experiments/ablation_feathers.sh $mode <device>' first." exception = e
    exit(1)
end

try
    global df_dphype_mnist_n250 = CSV.read("/workspace/dp-hype/algorithms/dphype_topk/dphype/results/$mode/dp-hype-ylecun-mnist-iid-N250-epsAll.csv", DataFrame)
    global df_dphype_cifar10_n250 = CSV.read("/workspace/dp-hype/algorithms/dphype_topk/dphype/results/$mode/dp-hype-uoft-cs-cifar10-iid-N250-epsAll.csv", DataFrame)
catch e
    @warn "Result file not found, plotting without dp-hype topk data. Execute 'experiments/privutility_tradeoff.sh $mode <device> 250' to include it." exception = e
    global df_dphype_mnist_n250 = DataFrame(eps=[0.1, 0.25, 0.5, 1.0, 3.0, Inf], accuracy=Float64[NaN, NaN, NaN, NaN, NaN, NaN])
    global df_dphype_cifar10_n250 = DataFrame(eps=[0.1, 0.25, 0.5, 1.0, 3.0, Inf], accuracy=Float64[NaN, NaN, NaN, NaN, NaN, NaN])
end

default(
    guidefont=font(12),
    tickfont=font(12),
    legendfont=font(12),
    titlefont=font(12),
    lw=2,
    markersize=6,
    markerstrokewidth=10,
)

markers = [:square, :circle, :utriangle, :diamond, :dtriangle,
    :hexagon, :star5, :cross, :xcross, :pentagon]

eps_inf_sub = 10.0

xticks = ([0.1, 0.25, 0.5, 1.0, 3.0, eps_inf_sub], ["0.1", "0.25", "0.5", "1.0", "3.0", "ꝏ"])

N = 250

p = plot(
    size=(600, 175),
    left_margin=5mm, right_margin=5mm, top_margin=5mm,
    bottom_margin=7mm,
    legend=:outerright,
)

ylims!(p, (0.0, 1.05))

data_plots = [df_feather_mnist_n250, df_feathers_cifar10_n250, df_dphype_mnist_n250, df_dphype_cifar10_n250]

for (i, df_a) in enumerate(data_plots)
    sort!(df_a, :eps, by=x -> x == "inf" ? Inf : x)
    df_a = transform(df_a, :eps => ByRow(x -> isfinite(x) ? x : eps_inf_sub) => :eps_plot)

    stats = combine(groupby(df_a, :eps_plot)) do g
        μ = mean(g.accuracy)
        σ = std(g.accuracy; corrected=false)
        N = nrow(g) + 2
        z = quantile(TDist(N - 1), 0.975)
        δ = z * σ / sqrt(N)
        (; eps_plot=g.eps_plot[1], mean=μ, lower=μ - δ, upper=μ + δ)
    end
    sort!(stats, :eps_plot)

    is_mnist = (i == 1 || i == 3)
    is_dphype = (i > 2)

    plot!(p, stats.eps_plot, stats.mean;
        ribbon=(stats.mean .- stats.lower, stats.upper .- stats.mean),
        label="",
        xscale=:log2,
        xticks=xticks,
        yticks=[0.2, 0.4, 0.6, 0.8, 1.0],
        marker=(is_mnist ? markers[2] : markers[1]),
        color=(is_dphype ? 2 : 1),
        linestyle=(is_dphype ? :solid : :dot),
        xlabel="Privacy Budget ε",
        ylabel="Accuracy",
    )

end

legend_specs = [
    ("DP-Hype, MNIST", markers[2], 2, :solid),
    ("Feathers, MNIST", markers[2], 1, :dot),
    ("DP-Hype, CIFAR10", markers[1], 2, :solid),
    ("Feathers, CIFAR10", markers[1], 1, :dot)
]

for (lbl, mkr, clr, ls) in legend_specs
    plot!(p, [NaN], [NaN]; label=lbl, marker=mkr, color=clr, linestyle=ls)
end


savefig("$save_dir/figure11_ablation_feathers.pdf")