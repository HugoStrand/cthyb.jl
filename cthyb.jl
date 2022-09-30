
using LinearAlgebra

struct Configuration
    t_i::Vector{Float64}
    t_f::Vector{Float64}
    function Configuration(t_i, t_f)
        @assert length(t_i) == length(t_f)
        #return length(t_i) == 0 ? new([], []) : new(sort(t_i), sort(t_f))
        return length(t_i) == 0 ? new([], []) : new(t_i, t_f)
    end
end

Base.:length(c::Configuration) = length(c.t_i)

struct Hybridization
    times::Vector{Float64}
    values::Vector{Float64}
end

function (Δ::Hybridization)(time::Float64)::Float64
    s = sign(time)
    time = mod(time, 1)
    idx = searchsortedfirst(Δ.times, time)
    ti, tf = Δ.times[idx-1], Δ.times[idx]
    vi, vf = Δ.values[idx-1], Δ.values[idx]
    return s * (vi + (time - ti) * (vf - vi) / (tf - ti))
end

function (Δ::Hybridization)(times::Vector{Float64})::Vector{Float64}
    return [ Δ(time) for time in times ]
end

struct Expansion
    β::Float64
    h::Float64
    Δ::Hybridization
end

struct Determinant
    mat::Array{Float64, 2}
    value::Float64

    function Determinant(c::Configuration, e::Expansion)
        mat = [ e.Δ(tf - ti) for tf in c.t_f, ti in c.t_i ]
        value = LinearAlgebra.det(mat)
        new(mat, value)
    end
end

struct Operator
    time::Float64
    operation::Int64
end

creation_operator(time::Float64) = Operator(time, +1)
annihilation_operator(time::Float64) = Operator(time, -1)

Base.:isless(o1::Operator, o2::Operator) = o1.time < o2.time

function configuration_operators(c::Configuration)
    creation_operators = [ creation_operator(time) for time in c.t_i ]
    annihilation_operators = [ annihilation_operator(time) for time in c.t_f ]
    sort(vcat(creation_operators, annihilation_operators))
end

function trace(c::Configuration, e::Expansion)
    if length(c) == 0
        return 1.0
    end
    ops = configuration_operators(c)
    first_state = first(ops).operation > 0 ? 0 : 1
    state = copy(first_state)
    t_i = 0.0

    value = (first_state == 1) ? 1.0 : -1.0
    #value = 1.0

    for op in ops
        t_f = op.time
        state += op.operation
        if state == 0
            continue
        elseif state == 1
            dt = t_f - t_i
            value *= exp(-e.h * dt)
        else
            value = 0
            break
        end
        t_i = t_f
    end
    if first_state != state
        return 0.0
    end
    return value
end

struct InsertMove
    t_i::Float64
    t_f::Float64
end

function Base.:(+)(c::Configuration, move::InsertMove)
    t_i = vcat(c.t_i, [move.t_i])
    t_f = vcat(c.t_f, [move.t_f])
    return Configuration(t_i, t_f)
end

function new_insert_move(c::Configuration, e::Expansion)
    t_i = e.β * rand()
    t_f = e.β * rand()
    return InsertMove(t_i, t_f)
end

function propose(move::InsertMove, c_old::Configuration, e::Expansion)
    c_new = c_old + move

    old = trace(c_old, e) * Determinant(c_old, e).value
    new = trace(c_new, e) * Determinant(c_new, e).value

    R = e.β^2 / length(c_new)^2 * new / old
    
    return R
end

function finalize(move::InsertMove, c::Configuration)
    c_new = c + move
    return c_new
end

struct RemovalMove
    i_idx::UInt64
    f_idx::UInt64
end

function Base.:(+)(c::Configuration, move::RemovalMove)
    if move.i_idx > 0
        t_i = deleteat!(copy(c.t_i), move.i_idx)
        t_f = deleteat!(copy(c.t_f), move.f_idx)
        return Configuration(t_i, t_f)
    else
        return c
    end
end

function new_removal_move(c::Configuration, e::Expansion)
    l = length(c)
    if l > 0
        i_idx = rand(1:l)
        f_idx = rand(1:l)
        return RemovalMove(i_idx, f_idx)
    else
        return RemovalMove(0, 0)
    end
end

function propose(move::RemovalMove, c_old::Configuration, e::Expansion)
    c_new = c_old + move

    old = trace(c_old, e) * Determinant(c_old, e).value
    new = trace(c_new, e) * Determinant(c_new, e).value

    R = length(c_old)^2 / e.β^2 * new / old
    
    return R
end

function finalize(move::RemovalMove, c::Configuration)
    c_new = c + move
    return c_new
end


function metropolis_hastings_update(c, e)

    # select move
    moves = [ new_insert_move, new_removal_move ]
    move_idx = rand(1:length(moves))
    move = moves[move_idx](c, e)
    #@show move
    
    R = propose(move, c, e)
    #R = R > 1 ? 1. : R
    #@show R
    
    # accept/reject move
    r = rand()
    if R > 1 || r < R
        c = finalize(move, c)
    end
    #@show length(c)
    
    return c
end

struct GreensFunction
    β::Float64
    data::Vector{Float64}
    function GreensFunction(β::Float64, data::Vector{Float64})
        new(β, data)
    end
    function GreensFunction(β::Float64, N::Int64)
        new(β, zeros(N))
    end
end

Base.:length(g::GreensFunction) = length(g.data)
Base.:(*)(g::GreensFunction, scalar::Float64) = GreensFunction(g.β, g.data * scalar)

function accumulate!(g::GreensFunction, time::Float64, value::Float64)
    if time < 0.0
        value *= -1
        time += g.β
    end
    @assert time >= 0.0
    @assert time <= g.β
    idx = ceil(Int, length(g) * time / g.β)
    g.data[idx] += value
end

function sample_greens_function!(g::GreensFunction, c::Configuration, e::Expansion)
    d = Determinant(c, e)
    M = LinearAlgebra.inv(d.mat)
    nt = length(c)
    for i in 1:nt, j in 1:nt
        t_i = c.t_i[i]
        t_j = c.t_f[j]
        #@show t_i, t_j
        accumulate!(g, t_j - t_i, M[j, i])
    end
end

β = 20.0
ϵ = 0.0
h = 0.0
V = 1.0

N_t = 100

times = range(0., β, N_t)
values = -V^2 * exp.(-ϵ .* times) / (1 + exp(-ϵ * β))
Δ = Hybridization(times, values)
#@show Δ

e = Expansion(β, h, Δ)

if false

    t_i = [0.1, 0.3, 0.5]
    t_f = [0.2, 0.4, 0.6]

    @show t_i
    @show t_f

    c = Configuration(t_i, t_f)
    @show c

    d = Determinant(c, e)
    @show d

    tr = trace(c, e)
    @show tr

    move = InsertMove(0.25, 0.29)
    @show move

    R = propose(move, c, e)
    @show R

    c = finalize(move, c)
    @show c

    move = RemovalMove(1, 1)
    @show move

    R = propose(move, c, e)
    @show R

    c = finalize(move, c)
    @show c

end

println("Starting CT-HYB QMC")

chunk = 100
warmup = 1000
sampling = 10000

nt = 100

c = Configuration([], [])
g = GreensFunction(β, nt)

println("Warmup sweeps $warmup with $chunk steps.")

for s in 1:warmup
    for i in 1:chunk
        global c
        c = metropolis_hastings_update(c, e)
    end    
end

@show length(c)
println("Sampling sweeps $sampling with $chunk steps.")

for s in 1:sampling
    for i in 1:chunk
        global c
        c = metropolis_hastings_update(c, e)
    end
    # sample
    #@show length(c)
    sample_greens_function!(g, c, e) 
end

g = g * (length(g)/sampling/β)

# Plot gf, cf exact result!
import PyPlot as plt

t = range(0.0, β, nt)
plt.plot(t, g.data, "-", label="G")
plt.plot(Δ.times, Δ.values, "-", label="Delta")
plt.legend(loc="best")
#plt.ylim([-1, 0])
plt.show()
