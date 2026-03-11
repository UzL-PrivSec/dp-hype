using Random
using Distributions
using ProgressBars
using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "--mode", "-m"
    help = "Mode: dry (10 iterations), artifacts (1000 iterations), paper (5000 iterations)"
    arg_type = String
end
parsed_args = parse_args(s)
mode = parsed_args["mode"]
if mode == "dry"
    num_iterations = 10
elseif mode == "artifacts"
    num_iterations = 1000
else
    num_iterations = 5000
end

μ_good = 1.0
μ_bad = 0.0
N = 250
eps_list = [1.0, 0.25]
k_list = [1, 5, 25, 50, 100]

sigmaeps_by_k = Dict{Int,Dict{Any,Float64}}(
    100 => Dict(
        1.0 => 55.0,
        0.25 => 205.0,
    ),
    50 => Dict(
        1.0 => 40.0,
        0.25 => 145.0,
    ),
    25 => Dict(
        1.0 => 27.0,
        0.25 => 100.3,
    ),
    10 => Dict(
        1.0 => 18.0,
        0.25 => 65.0,
    ),
    5 => Dict(
        1.0 => 12.5,
        0.25 => 46.0,
    ),
    3 => Dict(
        1.0 => 9.5,
        0.25 => 35.5,
    ),
    2 => Dict(
        1.0 => 8.0,
        0.25 => 29.1,
    ),
    1 => Dict(
        1.0 => 5.5,
        0.25 => 20.5,
    ),
)

function generate_utilities(H::Int, N::Int, h_abs::Int, μ_good::Float64, μ_bad::Float64, σ_user::Float64)
    idx_good = sample(1:H, h_abs; replace=false)
    U = zeros(Float64, N, H)
    noise = Normal(0, σ_user)
    for h in 1:H
        μ = (h in idx_good) ? μ_good : μ_bad
        @inbounds for n in 1:N
            U[n, h] = μ + rand(noise)
        end
    end
    return U, idx_good
end


function top_k_baseline(U, sigma::Float64, k::Int)

    local_tops = mapslices(x -> sortperm(x, rev=true), U; dims=2)[:, 1:k]

    noisy_votes = zeros(Float64, size(U, 2))
    for i in local_tops
        noisy_votes[i] += 1
    end
    noisy_votes .+= rand(Normal(0, sigma), size(noisy_votes))

    return argmax(noisy_votes)
end

data = "eps,σ_user,h_abs,k,num_hps,top_k_acc\n"

h_good_list = [1, 2, 4, 8, 16, 32, 64, 100]
σ_user_list = [0.2]
num_hps_list = [100]

for eps in eps_list
    for k in k_list
        for σ_user in σ_user_list
            for num_hps in num_hps_list
                for h_good in h_good_list
                    sigma = sigmaeps_by_k[k][eps]
                    println("Running for ϵ=$eps, σ_user=$σ_user, h_abs=$h_good, k=$k, σ=$sigma, num_hps=$num_hps")
                    acc_top_k = 0
                    for _ in ProgressBar(1:num_iterations)
                        U, idx_good = generate_utilities(num_hps, N, h_good, μ_good, μ_bad, σ_user)
                        goodset = Set(idx_good)

                        w_top_k = top_k_baseline(U, sigma, k)

                        acc_top_k += (w_top_k in goodset) ? 1 : 0
                    end
                    println("Top-$k Acc: ", acc_top_k / num_iterations)
                    global data *= join([string(eps), string(σ_user), string(h_good), string(k), string(num_hps), string(acc_top_k / num_iterations)], ",") * "\n"
                end
            end
        end
    end
end

open("/workspace/dp-hype/algorithms/simulation/results/$mode/sim_hgood.csv", "w") do io
    write(io, data)
end