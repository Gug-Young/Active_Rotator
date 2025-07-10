using OrdinaryDiffEq
using CuArrays
CuArrays.allowscalar(false)

function func(du, u, p, t)
    @inbounds @. du = u
    return nothing
end


tspan = (0f0, 10f0)
alg = BS3()
u0 = fill(1f0, Int(5e5))

# ******************************************************************************
println("CPU:")
prob = ODEProblem(func, u0, tspan)

@time sol = solve(prob, alg, saveat=1f-2)
@time sol = solve(prob, alg, saveat=1f-2)
@time sol = solve(prob, alg, saveat=1f-2)


# ******************************************************************************
println("GPU:")

# CuArrays.allowscalar(false)

u0_gpu = CuArray(convert(Array{Float32, 1}, u0))
prob_gpu = ODEProblem(func, u0_gpu, tspan)

sol_gpu = nothing; GC.gc(true); CuArrays.reclaim()
@time sol_gpu = solve(prob_gpu, alg, saveat=1f-2)
sol_gpu = nothing; GC.gc(true); CuArrays.reclaim()
@time sol_gpu = solve(prob_gpu, alg, saveat=1f-2)
sol_gpu = nothing; GC.gc(true); CuArrays.reclaim()
@time sol_gpu = solve(prob_gpu, alg, saveat=1f-2)