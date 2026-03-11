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

base_dir = "/workspace/dp-hype/algorithms/dphype_topk/dphype/results/$mode"
path_opt_data = "/workspace/dp-hype/opt_data"
save_dir = "/workspace/dp-hype/figures/$mode"

try
    global df_mnist_a05_N50 = CSV.read("$base_dir/opt-ylecun-mnist-dirichlet-0.5-N50-epsAll.csv", DataFrame)
    global df_mnist_a05_N100 = CSV.read("$base_dir/opt-ylecun-mnist-dirichlet-0.5-N100-epsAll.csv", DataFrame)
    global df_mnist_a05_N250 = CSV.read("$base_dir/opt-ylecun-mnist-dirichlet-0.5-N250-epsAll.csv", DataFrame)
    global df_mnist_a5_N50 = CSV.read("$base_dir/opt-ylecun-mnist-dirichlet-5.0-N50-epsAll.csv", DataFrame)
    global df_mnist_a5_N100 = CSV.read("$base_dir/opt-ylecun-mnist-dirichlet-5.0-N100-epsAll.csv", DataFrame)
    global df_mnist_a5_N250 = CSV.read("$base_dir/opt-ylecun-mnist-dirichlet-5.0-N250-epsAll.csv", DataFrame)
    global df_mnist_a30_N50 = CSV.read("$base_dir/opt-ylecun-mnist-dirichlet-30.0-N50-epsAll.csv", DataFrame)
    global df_mnist_a30_N100 = CSV.read("$base_dir/opt-ylecun-mnist-dirichlet-30.0-N100-epsAll.csv", DataFrame)
    global df_mnist_a30_N250 = CSV.read("$base_dir/opt-ylecun-mnist-dirichlet-30.0-N250-epsAll.csv", DataFrame)
    global df_cifar10_a05_N50 = CSV.read("$base_dir/opt-uoft-cs-cifar10-dirichlet-0.5-N50-epsAll.csv", DataFrame)
    global df_cifar10_a05_N100 = CSV.read("$base_dir/opt-uoft-cs-cifar10-dirichlet-0.5-N100-epsAll.csv", DataFrame)
    global df_cifar10_a05_N250 = CSV.read("$base_dir/opt-uoft-cs-cifar10-dirichlet-0.5-N250-epsAll.csv", DataFrame)
    global df_cifar10_a5_N50 = CSV.read("$base_dir/opt-uoft-cs-cifar10-dirichlet-5.0-N50-epsAll.csv", DataFrame)
    global df_cifar10_a5_N100 = CSV.read("$base_dir/opt-uoft-cs-cifar10-dirichlet-5.0-N100-epsAll.csv", DataFrame)
    global df_cifar10_a5_N250 = CSV.read("$base_dir/opt-uoft-cs-cifar10-dirichlet-5.0-N250-epsAll.csv", DataFrame)
    global df_cifar10_a30_N50 = CSV.read("$base_dir/opt-uoft-cs-cifar10-dirichlet-30.0-N50-epsAll.csv", DataFrame)
    global df_cifar10_a30_N100 = CSV.read("$base_dir/opt-uoft-cs-cifar10-dirichlet-30.0-N100-epsAll.csv", DataFrame)
    global df_cifar10_a30_N250 = CSV.read("$base_dir/opt-uoft-cs-cifar10-dirichlet-30.0-N250-epsAll.csv", DataFrame)
    global df_adult_a05_N50 = CSV.read("$base_dir/opt-scikit-learn-adult-census-income-dirichlet-0.5-N50-epsAll.csv", DataFrame)
    global df_adult_a05_N100 = CSV.read("$base_dir/opt-scikit-learn-adult-census-income-dirichlet-0.5-N100-epsAll.csv", DataFrame)
    global df_adult_a05_N250 = CSV.read("$base_dir/opt-scikit-learn-adult-census-income-dirichlet-2.0-N250-epsAll.csv", DataFrame)
    global df_adult_a5_N50 = CSV.read("$base_dir/opt-scikit-learn-adult-census-income-dirichlet-5.0-N50-epsAll.csv", DataFrame)
    global df_adult_a5_N100 = CSV.read("$base_dir/opt-scikit-learn-adult-census-income-dirichlet-5.0-N100-epsAll.csv", DataFrame)
    global df_adult_a5_N250 = CSV.read("$base_dir/opt-scikit-learn-adult-census-income-dirichlet-5.0-N250-epsAll.csv", DataFrame)
    global df_adult_a30_N50 = CSV.read("$base_dir/opt-scikit-learn-adult-census-income-dirichlet-30.0-N50-epsAll.csv", DataFrame)
    global df_adult_a30_N100 = CSV.read("$base_dir/opt-scikit-learn-adult-census-income-dirichlet-30.0-N100-epsAll.csv", DataFrame)
    global df_adult_a30_N250 = CSV.read("$base_dir/opt-scikit-learn-adult-census-income-dirichlet-30.0-N250-epsAll.csv", DataFrame)
catch e
    @warn "Opt result file not found -> Falling back to /workspace/dp-hype/opt_data/. Or execute 'experiments/privutility_tradeoff.sh $mode <device> 100 yes' first" exception = e
    try
        global df_mnist_a05_N50 = CSV.read("$path_opt_data/opt-ylecun-mnist-dirichlet-0.5-N50-epsAll.csv", DataFrame)
        global df_mnist_a05_N100 = CSV.read("$path_opt_data/opt-ylecun-mnist-dirichlet-0.5-N100-epsAll.csv", DataFrame)
        global df_mnist_a05_N250 = CSV.read("$path_opt_data/opt-ylecun-mnist-dirichlet-0.5-N250-epsAll.csv", DataFrame)
        global df_mnist_a5_N50 = CSV.read("$path_opt_data/opt-ylecun-mnist-dirichlet-5.0-N50-epsAll.csv", DataFrame)
        global df_mnist_a5_N100 = CSV.read("$path_opt_data/opt-ylecun-mnist-dirichlet-5.0-N100-epsAll.csv", DataFrame)
        global df_mnist_a5_N250 = CSV.read("$path_opt_data/opt-ylecun-mnist-dirichlet-5.0-N250-epsAll.csv", DataFrame)
        global df_mnist_a30_N50 = CSV.read("$path_opt_data/opt-ylecun-mnist-dirichlet-30.0-N50-epsAll.csv", DataFrame)
        global df_mnist_a30_N100 = CSV.read("$path_opt_data/opt-ylecun-mnist-dirichlet-30.0-N100-epsAll.csv", DataFrame)
        global df_mnist_a30_N250 = CSV.read("$path_opt_data/opt-ylecun-mnist-dirichlet-30.0-N250-epsAll.csv", DataFrame)
        global df_cifar10_a05_N50 = CSV.read("$path_opt_data/opt-uoft-cs-cifar10-dirichlet-0.5-N50-epsAll.csv", DataFrame)
        global df_cifar10_a05_N100 = CSV.read("$path_opt_data/opt-uoft-cs-cifar10-dirichlet-0.5-N100-epsAll.csv", DataFrame)
        global df_cifar10_a05_N250 = CSV.read("$path_opt_data/opt-uoft-cs-cifar10-dirichlet-0.5-N250-epsAll.csv", DataFrame)
        global df_cifar10_a5_N50 = CSV.read("$path_opt_data/opt-uoft-cs-cifar10-dirichlet-5.0-N50-epsAll.csv", DataFrame)
        global df_cifar10_a5_N100 = CSV.read("$path_opt_data/opt-uoft-cs-cifar10-dirichlet-5.0-N100-epsAll.csv", DataFrame)
        global df_cifar10_a5_N250 = CSV.read("$path_opt_data/opt-uoft-cs-cifar10-dirichlet-5.0-N250-epsAll.csv", DataFrame)
        global df_cifar10_a30_N50 = CSV.read("$path_opt_data/opt-uoft-cs-cifar10-dirichlet-30.0-N50-epsAll.csv", DataFrame)
        global df_cifar10_a30_N100 = CSV.read("$path_opt_data/opt-uoft-cs-cifar10-dirichlet-30.0-N100-epsAll.csv", DataFrame)
        global df_cifar10_a30_N250 = CSV.read("$path_opt_data/opt-uoft-cs-cifar10-dirichlet-30.0-N250-epsAll.csv", DataFrame)
        global df_adult_a05_N50 = CSV.read("$path_opt_data/opt-scikit-learn-adult-census-income-dirichlet-0.5-N50-epsAll.csv", DataFrame)
        global df_adult_a05_N100 = CSV.read("$path_opt_data/opt-scikit-learn-adult-census-income-dirichlet-0.5-N100-epsAll.csv", DataFrame)
        global df_adult_a05_N250 = CSV.read("$path_opt_data/opt-scikit-learn-adult-census-income-dirichlet-2.0-N250-epsAll.csv", DataFrame)
        global df_adult_a5_N50 = CSV.read("$path_opt_data/opt-scikit-learn-adult-census-income-dirichlet-5.0-N50-epsAll.csv", DataFrame)
        global df_adult_a5_N100 = CSV.read("$path_opt_data/opt-scikit-learn-adult-census-income-dirichlet-5.0-N100-epsAll.csv", DataFrame)
        global df_adult_a5_N250 = CSV.read("$path_opt_data/opt-scikit-learn-adult-census-income-dirichlet-5.0-N250-epsAll.csv", DataFrame)
        global df_adult_a30_N50 = CSV.read("$path_opt_data/opt-scikit-learn-adult-census-income-dirichlet-30.0-N50-epsAll.csv", DataFrame)
        global df_adult_a30_N100 = CSV.read("$path_opt_data/opt-scikit-learn-adult-census-income-dirichlet-30.0-N100-epsAll.csv", DataFrame)
        global df_adult_a30_N250 = CSV.read("$path_opt_data/opt-scikit-learn-adult-census-income-dirichlet-30.0-N250-epsAll.csv", DataFrame)
    catch e
        @error "Ooops something went wrong." exception = e
        exit(1)
    end
end

df_mnist_a05_N50.alpha .= 0.5
df_mnist_a05_N100.alpha .= 0.5
df_mnist_a05_N250.alpha .= 0.5
df_mnist_a05_N50.N .= 50
df_mnist_a05_N100.N .= 100
df_mnist_a05_N250.N .= 250
df_mnist_a5_N50.alpha .= 5.0
df_mnist_a5_N100.alpha .= 5.0
df_mnist_a5_N250.alpha .= 5.0
df_mnist_a5_N50.N .= 50
df_mnist_a5_N100.N .= 100
df_mnist_a5_N250.N .= 250
df_mnist_a30_N50.alpha .= 30.0
df_mnist_a30_N100.alpha .= 30.0
df_mnist_a30_N250.alpha .= 30.0
df_mnist_a30_N50.N .= 50
df_mnist_a30_N100.N .= 100
df_mnist_a30_N250.N .= 250

df_cifar10_a05_N50.alpha .= 0.5
df_cifar10_a05_N100.alpha .= 0.5
df_cifar10_a05_N250.alpha .= 0.5
df_cifar10_a05_N50.N .= 50
df_cifar10_a05_N100.N .= 100
df_cifar10_a05_N250.N .= 250
df_cifar10_a5_N50.alpha .= 5.0
df_cifar10_a5_N100.alpha .= 5.0
df_cifar10_a5_N250.alpha .= 5.0
df_cifar10_a5_N50.N .= 50
df_cifar10_a5_N100.N .= 100
df_cifar10_a5_N250.N .= 250
df_cifar10_a30_N50.alpha .= 30.0
df_cifar10_a30_N100.alpha .= 30.0
df_cifar10_a30_N250.alpha .= 30.0
df_cifar10_a30_N50.N .= 50
df_cifar10_a30_N100.N .= 100
df_cifar10_a30_N250.N .= 250

df_adult_a05_N50.alpha .= 0.5
df_adult_a05_N100.alpha .= 0.5
df_adult_a05_N250.alpha .= 0.5
df_adult_a05_N50.N .= 50
df_adult_a05_N100.N .= 100
df_adult_a05_N250.N .= 250
df_adult_a5_N50.alpha .= 5.0
df_adult_a5_N100.alpha .= 5.0
df_adult_a5_N250.alpha .= 5.0
df_adult_a5_N50.N .= 50
df_adult_a5_N100.N .= 100
df_adult_a5_N250.N .= 250
df_adult_a30_N50.alpha .= 30.0
df_adult_a30_N100.alpha .= 30.0
df_adult_a30_N250.alpha .= 30.0
df_adult_a30_N50.N .= 50
df_adult_a30_N100.N .= 100
df_adult_a30_N250.N .= 250

df_mnist = vcat(df_mnist_a05_N50, df_mnist_a05_N100, df_mnist_a05_N250, df_mnist_a5_N50, df_mnist_a5_N100, df_mnist_a5_N250, df_mnist_a30_N50, df_mnist_a30_N100, df_mnist_a30_N250)

df_cifar10 = vcat(df_cifar10_a05_N50, df_cifar10_a05_N100, df_cifar10_a05_N250, df_cifar10_a5_N50, df_cifar10_a5_N100, df_cifar10_a5_N250, df_cifar10_a30_N50, df_cifar10_a30_N100, df_cifar10_a30_N250)

df_adult = vcat(df_adult_a05_N50, df_adult_a05_N100, df_adult_a05_N250, df_adult_a5_N50, df_adult_a5_N100, df_adult_a5_N250, df_adult_a30_N50, df_adult_a30_N100, df_adult_a30_N250)

default(
    guidefont=font(12),
    tickfont=font(12),
    legendfont=font(12),
    titlefont=font(12),
    lw=1,
    markersize=6,
    markerstrokewidth=10,
)

num_bins = 50

# alpha = 0.5

lay = @layout([a b c])

p = plot(
    layout=lay,
    size=(1050, 175),
    left_margin=5mm, right_margin=5mm, top_margin=5mm,
    bottom_margin=10mm,
    legend=false,
)

xlims!(p, (0.0, 1.0))
ylims!(p, (0.0, 50.0))

for (i, n) in enumerate([50, 100, 250])
    dtndf = df_mnist[(df_mnist.N.==n).&(df_mnist.alpha.==0.5), :]
    @df dtndf histogram!(p, subplot=(4 - i), :accuracy, bins=range(0.0, 1.0, num_bins), label="", alpha=0.5, ylabel=(i == 3 ? "Frequency" : ""),
        xlabel="Accuracy"
    )
    dtndf = df_cifar10[(df_cifar10.N.==n).&(df_cifar10.alpha.==0.5), :]
    @df dtndf histogram!(p, subplot=(4 - i), :accuracy, bins=bins = range(0.0, 1.0, num_bins), label="", alpha=0.5,
    )
    dtndf = df_adult[(df_adult.N.==n).&(df_adult.alpha.==0.5), :]
    @df dtndf histogram!(p, subplot=(4 - i), :accuracy, bins=bins = range(0.0, 1.0, num_bins), label="", alpha=0.5,
    )
end

savefig("$save_dir/figure8c_acc_distribution_noniid_a05allN.pdf")

# alpha = 5.0

lay = @layout([a b c])

p = plot(
    layout=lay,
    size=(1050, 150),
    left_margin=5mm, right_margin=5mm, top_margin=5mm,
    bottom_margin=5mm,            # small overall bottom margin
    legend=false,             # no per-subplot legends
)

xlims!(p, (0.0, 1.0))
ylims!(p, (0.0, 50.0))

for (i, n) in enumerate([50, 100, 250])
    dtndf = df_mnist[(df_mnist.N.==n).&(df_mnist.alpha.==5.0), :]
    @df dtndf histogram!(p, subplot=(4 - i), :accuracy, bins=range(0.0, 1.0, num_bins), label="", alpha=0.5, ylabel=(i == 3 ? "Frequency" : "")
    )
    dtndf = df_cifar10[(df_cifar10.N.==n).&(df_cifar10.alpha.==5.0), :]
    @df dtndf histogram!(p, subplot=(4 - i), :accuracy, bins=bins = range(0.0, 1.0, num_bins), label="", alpha=0.5,
    )
    dtndf = df_adult[(df_adult.N.==n).&(df_adult.alpha.==5.0), :]
    @df dtndf histogram!(p, subplot=(4 - i), :accuracy, bins=bins = range(0.0, 1.0, num_bins), label="", alpha=0.5,
    )
end

savefig("$save_dir/figure8b_acc_distribution_noniid_a5allN.pdf")

# alpha = 30.0

lay = @layout([a b c])

p = plot(
    layout=lay,
    size=(1050, 150),
    left_margin=5mm, right_margin=5mm, top_margin=5mm,
    bottom_margin=5mm,            # small overall bottom margin
    legend=false,             # no per-subplot legends
)

xlims!(p, (0.0, 1.0))
ylims!(p, (0.0, 50.0))

for (i, n) in enumerate([50, 100, 250])
    dtndf = df_mnist[(df_mnist.N.==n).&(df_mnist.alpha.==30.0), :]
    @df dtndf histogram!(p, subplot=(4 - i), :accuracy, bins=range(0.0, 1.0, num_bins), label="", alpha=0.5, ylabel=(i == 3 ? "Frequency" : ""),
        title="$n Clients"
    )
    dtndf = df_cifar10[(df_cifar10.N.==n).&(df_cifar10.alpha.==30.0), :]
    @df dtndf histogram!(p, subplot=(4 - i), :accuracy, bins=bins = range(0.0, 1.0, num_bins), label="", alpha=0.5, title="$n Clients"
    )
    dtndf = df_adult[(df_adult.N.==n).&(df_adult.alpha.==30.0), :]
    @df dtndf histogram!(p, subplot=(4 - i), :accuracy, bins=bins = range(0.0, 1.0, num_bins), label="", alpha=0.5, title="$n Clients"
    )
end

savefig("$save_dir/figure8a_acc_distribution_noniid_a30allN.pdf")

# legend

p = plot(
    size=(1050, 75),
    left_margin=5mm, right_margin=5mm, top_margin=5mm,
    bottom_margin=5mm,            # small overall bottom margin
)

for (dt_name, linestyle) in zip(["MNIST", "CIFAR-10", "Adult"], [:solid, :dot, :dashdotdot])
    plot!(p, [NaN], [NaN];
        #marker=markers[k],
        #seriescolor=k_index[k],
        # linestyle=linestyle,
        label=dt_name)
end

plot!(p;
    legend=:bottom,               # center within the full-width cell
    legend_column=-1,             # horizontal legend
    legend_title="Data Sets",
    xaxis=false, yaxis=false, grid=false, framestyle=:none)

savefig("$save_dir/figure8_acc_distribution_noniid_legend.pdf")

