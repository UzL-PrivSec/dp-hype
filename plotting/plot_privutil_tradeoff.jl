using StatsPlots
using CSV
using DataFrames
using Measures
using LaTeXStrings
using Statistics
using Distributions
using ArgParse

s = ArgParseSettings()

@add_arg_table s begin
    "-n"
    arg_type = Int
    required = true
    "-d"
    arg_type = String
    required = true
    "--non_iid"
    action = :store_true
    "-a"
    arg_type = Float64
    required = true
    "--mode", "-m"
    help = "Mode: dry (10 iterations), artifacts (1000 iterations), paper (5000 iterations)"
    arg_type = String
    required = true
end

args = parse_args(s)

println(args)

valid_ns = [50, 100, 250]
valid_datasets = ["ylecun/mnist", "uoft-cs/cifar10", "scikit-learn/adult-census-income"]
valid_alphas = [0.5, 2.0, 5.0, 30.0]
valid_modes = ["dry", "artifacts", "paper"]

if args["n"] ∉ valid_ns
    @error "Invalid -n value '$(args["n"])'. Must be one of: $(join(valid_ns, ", "))"
    exit(1)
end
if args["d"] ∉ valid_datasets
    @error "Invalid -d value '$(args["d"])'. Must be one of: $(join(valid_datasets, ", "))"
    exit(1)
end
if args["non_iid"] && args["a"] ∉ valid_alphas
    @error "Invalid -a value '$(args["a"])'. Must be one of: $(join(valid_alphas, ", "))"
    exit(1)
end
if args["non_iid"] && args["a"] == 2.0 && args["n"] == 250 && args["d"] != "scikit-learn/adult-census-income"
    @error "Only the Adult data set is allowed to have alpha=2.0 for n=250."
    exit(1)
end
if args["mode"] ∉ valid_modes
    @error "Invalid -m value '$(args["mode"])'. Must be one of: $(join(valid_modes, ", "))"
    exit(1)
end

base_dir = "/workspace/dp-hype/algorithms/dphype_topk/dphype/results/$(args["mode"])/"
base_dir_opt_data = "/workspace/dp-hype/opt_data/"
base_save_dir = "/workspace/dp-hype/figures/$(args["mode"])/"

iid_scenario_str = args["non_iid"] ? "dirichlet-$(args["a"])" : "iid"
result_name = "$(replace(args["d"], "/" => "-"))-$(iid_scenario_str)-N$(args["n"])-epsAll.csv"
path_dphype = base_dir * "dp-hype-$(result_name)"
path_opt_data = base_dir_opt_data * "opt-$(result_name)"
path_opt = base_dir * "opt-$(result_name)"

println(path_dphype)
println(path_opt)

n_to_figurename = Dict(
    50 => "figure13",
    100 => "figure5",
    250 => "figure12",
)


if args["non_iid"]
    figure_prefix = n_to_figurename[args["n"]]
else
    figure_prefix = "figure4"
end

dataset_translations = Dict(
    "ylecun/mnist" => "MNIST",
    "uoft-cs/cifar10" => "CIFAR-10",
    "scikit-learn/adult-census-income" => "Adult",
)
dataset_name = dataset_translations[args["d"]]

try
    global df_opt = CSV.read(path_opt, DataFrame)
catch e
    @warn "Opt result file not found -> Falling back to /workspace/dp-hype/opt_data/. Or execute 'experiments/privutility_tradeoff.sh $(args["mode"]) <device> $(args["n"]) yes' first" exception = e
    try
        global df_opt = CSV.read(path_opt_data, DataFrame)
    catch e
        @error "Ooops something went wrong." exception = e
        exit(1)
    end
end

try
    global df_dphype = CSV.read(path_dphype, DataFrame)
catch e
    @error "Result file not found. Please execute 'experiments/privutility_tradeoff.sh $(args["mode"]) <device> $(args["n"]) no' first." exception = e
    exit(1)
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

markers = [:circle, :square, :utriangle, :diamond, :dtriangle,
    :hexagon, :star5, :cross, :xcross, :pentagon]

eps_inf_sub = 10.0

xticks = ([0.1, 0.25, 0.5, 1.0, 3.0, eps_inf_sub], ["0.1", "0.25", "0.5", "1.0", "3.0", "ꝏ"])

p = plot(
    size=(350, 200),
    left_margin=5mm, right_margin=5mm, top_margin=5mm,
    bottom_margin=4mm,
    legend=false,
)

ylims_by_dataset = Dict(
    "ylecun/mnist" => (0.1, 1.04),
    "uoft-cs/cifar10" => (args["non_iid"] ? (0.0, 0.62) : (0.0, 0.8)),
    "scikit-learn/adult-census-income" => (0.45, 0.9),
)

ylims!(p, ylims_by_dataset[args["d"]])

opt_acc = maximum(df_opt.accuracy)

sort!(df_dphype, :eps, by=x -> x == "inf" ? Inf : x)
df_dphype = transform(df_dphype, :eps => ByRow(x -> isfinite(x) ? x : eps_inf_sub) => :eps_plot)

stats = combine(groupby(df_dphype, :eps_plot)) do g
    μ = mean(g.accuracy)
    σ = std(g.accuracy; corrected=false)
    N = nrow(g)
    z = quantile(TDist(args["n"] - 1), 0.975)   # 95% CI
    δ = z * σ / sqrt(args["n"])
    (; eps_plot=g.eps_plot[1], mean=μ, lower=μ - δ, upper=μ + δ)
end
sort!(stats, :eps_plot)

plot!(p, stats.eps_plot, stats.mean;
    ribbon=(stats.mean .- stats.lower, stats.upper .- stats.mean),
    xscale=:log2,
    xticks=xticks,
    marker=markers[1],
    title="n=$(args["n"]),$(args["non_iid"] ? "beta-$(args["a"])" : "iid"),$(dataset_name)",
    xlabel="Privacy Budget ε",
    ylabel="Accuracy",
)

μ_randg = mean(df_opt.accuracy)

plot!(p, [0.1, 0.25, 0.5, 1.0, 3.0, eps_inf_sub], fill(μ_randg, 6);
    label="RandGuess",
    xscale=:log2,
    marker=markers[2],
    linestyle=:dash,
)

plot!(p, [0.1, 0.25, 0.5, 1.0, 3.0, eps_inf_sub], fill(opt_acc, 6);
    label="Opt",
    xscale=:log2,
    marker=markers[3],
    linestyle=:dot,
)

savefig(base_save_dir * "$(figure_prefix)_privutil_tradeoff_$(dataset_name)_N$(args["n"])_$(iid_scenario_str).pdf")

# legend

legend_path = base_save_dir * "$(figure_prefix)_privutil_tradeoff_legend.pdf"
if isfile(legend_path)
    exit(0)
end

p = plot(
    size=(1050, 75),
    left_margin=5mm, right_margin=5mm, top_margin=5mm,
    bottom_margin=2mm,            # small overall bottom margin
    legend=false,             # no per-subplot legends
)

for (i, algo) in enumerate([("DP-Hype", :solid), ("RandGuess", :dash), ("Opt", :dot)])
    plot!(p, [NaN], [NaN];
        # top_margin=13mm,
        marker=markers[i],
        linestyle=algo[2],
        label=algo[1])
end

plot!(p;
    legend=:bottom,
    legend_column=-1,
    legend_title="Algorithms",
    xaxis=false, yaxis=false, grid=false, framestyle=:none)

savefig(legend_path)
