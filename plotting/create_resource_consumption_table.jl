using CSV
using DataFrames
using Statistics
using Printf
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

mean_skip(v) = mean(skipmissing(v))
bytes_to_mib(x) = x / 1024^2
fmt2(x) = (x === missing || x === nothing) ? "" : @sprintf("%.2f", x)

base_dir = "/workspace/dp-hype/algorithms/dphype_topk/dphype/results/$mode"
save_dir = "/workspace/dp-hype/plotting/$mode"

try
    global df_fv = CSV.read("/workspace/dp-hype/algorithms/secsum_monitor/results/$mode/bandwidth_results.csv", DataFrame)

    global df_lhp_mnist_50 = CSV.read("$base_dir/client_HP_eval_resstats_ylecun-mnist_N50.csv", DataFrame)
    global df_lhp_mnist_100 = CSV.read("$base_dir/client_HP_eval_resstats_ylecun-mnist_N100.csv", DataFrame)
    global df_lhp_mnist_250 = CSV.read("$base_dir/client_HP_eval_resstats_ylecun-mnist_N250.csv", DataFrame)
    global df_lhp_cifar_50 = CSV.read("$base_dir/client_HP_eval_resstats_uoft-cs-cifar10_N50.csv", DataFrame)
    global df_lhp_cifar_100 = CSV.read("$base_dir/client_HP_eval_resstats_uoft-cs-cifar10_N100.csv", DataFrame)
    global df_lhp_cifar_250 = CSV.read("$base_dir/client_HP_eval_resstats_uoft-cs-cifar10_N250.csv", DataFrame)
    global df_lhp_adult_50 = CSV.read("$base_dir/client_HP_eval_resstats_scikit-learn-adult-census-income_N50.csv", DataFrame)
    global df_lhp_adult_100 = CSV.read("$base_dir/client_HP_eval_resstats_scikit-learn-adult-census-income_N100.csv", DataFrame)
    global df_lhp_adult_250 = CSV.read("$base_dir/client_HP_eval_resstats_scikit-learn-adult-census-income_N250.csv", DataFrame)
catch e
    @error "Result file not found. Please execute 'experiments/resource_consumption.sh $mode' first." exception = e
    exit(1)
end

df_lhp_mnist_50.dataset .= "MNIST"
df_lhp_mnist_100.dataset .= "MNIST"
df_lhp_mnist_250.dataset .= "MNIST"

df_lhp_cifar_50.dataset .= "Cifar-10"
df_lhp_cifar_100.dataset .= "Cifar-10"
df_lhp_cifar_250.dataset .= "Cifar-10"

df_lhp_adult_50.dataset .= "Adult"
df_lhp_adult_100.dataset .= "Adult"
df_lhp_adult_250.dataset .= "Adult"

lhp_entries = [
    (50, "MNIST", df_lhp_mnist_50),
    (50, "Cifar-10", df_lhp_cifar_50),
    (50, "Adult", df_lhp_adult_50),
    (100, "MNIST", df_lhp_mnist_100),
    (100, "Cifar-10", df_lhp_cifar_100),
    (100, "Adult", df_lhp_adult_100),
    (250, "MNIST", df_lhp_mnist_250),
    (250, "Cifar-10", df_lhp_cifar_250),
    (250, "Adult", df_lhp_adult_250),
]

fv_g = combine(groupby(df_fv, [:num_clients]),
    :avg_upload_bytes => mean_skip => :upload_bytes,
    :avg_download_bytes => mean_skip => :download_bytes,
    :runtime_seconds => mean_skip => :fv_rt_s,
)
fv_g.upload_mib = fv_g.upload_bytes ./ 1024
fv_g.download_mib = fv_g.download_bytes ./ 1024

struct TableRow
    n::Int
    dataset::String
    vram::Float64
    lhp_rt::Float64
    bw_up::Float64
    bw_down::Float64
    fv_rt::Float64
end

table_rows = TableRow[]

for (n, dataset, df_lhp) in lhp_entries
    vram = mean_skip(df_lhp.peak_gpu_memory_mb)
    lhp_rt = mean_skip(df_lhp.time_seconds)
    fv_row = only(filter(r -> r.num_clients == n, fv_g))
    push!(table_rows, TableRow(n, dataset, vram, lhp_rt, fv_row.upload_mib, fv_row.download_mib, fv_row.fv_rt_s))
end

ns = unique([r.n for r in table_rows])

function build_latex(table_rows, ns)
    lines = String[]
    push!(lines, raw"\documentclass{article}")
    push!(lines, raw"\usepackage{booktabs}")
    push!(lines, raw"\usepackage{multirow}")
    push!(lines, raw"\usepackage[margin=2cm]{geometry}")
    push!(lines, raw"\begin{document}")
    push!(lines, raw"\begin{table}[h]")
    push!(lines, raw"\centering")
    push!(lines, raw"\begin{tabular}{clrr rr}")
    push!(lines, raw"\toprule")
    push!(lines, raw"& & \multicolumn{2}{c}{Local HP Evaluation} & \multicolumn{2}{c}{Federated Voting} \\ \cmidrule(lr){3-4}\cmidrule(l){5-6}")
    push!(lines, "\$n\$ & Data set & vRAM & RT & Bandwidth & RT \\\\")
    push!(lines, raw"\midrule")

    for (gi, n) in enumerate(ns)
        group = filter(r -> r.n == n, table_rows)
        for (i, r) in enumerate(group)
            n_cell = i == 1 ? "\\multirow{$(length(group))}{*}{$n}" : ""
            if i == div(length(group), 2) + 1
                bw = "$(fmt2(r.bw_up)) / $(fmt2(r.bw_down))"
                fvrt = fmt2(r.fv_rt)
            else
                bw = ""
                fvrt = ""
            end
            push!(lines, "$n_cell & $(r.dataset) & $(fmt2(r.vram)) & $(fmt2(r.lhp_rt)) & $bw & $fvrt \\\\")
        end
        if gi < length(ns)
            push!(lines, raw"\midrule")
        end
    end

    push!(lines, raw"\bottomrule")
    push!(lines, raw"\end{tabular}")
    push!(lines, raw"\end{table}")
    push!(lines, raw"\end{document}")
    return join(lines, "\n")
end

latex_str = build_latex(table_rows, ns)

temp_dir = "/workspace/dp-hype/.temp"

open("$temp_dir/table4_resource_consumption_table.tex", "w") do f
    write(f, latex_str)
end

run(ignorestatus(`pdflatex -interaction=nonstopmode -output-directory=$temp_dir $temp_dir/table4_resource_consumption_table.tex`))