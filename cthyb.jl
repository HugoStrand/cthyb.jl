raw"""
Segment picture hybridization expansion

Author: Hugo U.R. Strand (2022)

"""

import PyCall
using LinearAlgebra

struct Configuration

    t_i::Vector{Float64}
    t_f::Vector{Float64}
    
    function Configuration(t_i, t_f)
        @assert length(t_i) == length(t_f)
        if length(t_i) == 0
            return new([], [])
        end

        t_i = sort(t_i)
        t_f = sort(t_f)

        if first(t_f) < first(t_i)
            push!(t_f, popfirst!(t_f))
        end

        return new(t_i, t_f)
    end
end

Base.:length(c::Configuration) = length(c.t_i)

struct Hybridization
    times::Vector{Float64}
    values::Vector{Float64}
    β::Float64
end

function (Δ::Hybridization)(time::Float64)::Float64

    s = 1.0
    if time < 0.0
        s = -1.0
        time += Δ.β
    end

    idx = searchsortedfirst(Δ.times, time)
    idx = idx == 1 ? 2 : idx
    
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

function trace(c::Configuration, e::Expansion)::Float64

    if length(c) == 0
        return 2.0
    end

    ops = configuration_operators(c)
    first_state = first(ops).operation > 0 ? 0 : 1    
    state = copy(first_state)

    t_i = 0.0
    value = (first_state == 1) ? -1.0 : +1.0
    
    for op in ops
        if state == 1
            value *= exp(-e.h * (op.time - t_i))
        end
        
        t_i = op.time
        state += op.operation

        if state < 0 || state > 1
            return 0.0
        end
    end

    if first_state != state
        return 0.0
    end

    if state == 1
        value *= exp(-e.h * (e.β - t_i))
    end

    return value
end

function eval(c::Configuration, e::Expansion)
    return trace(c, e) * Determinant(c, e).value
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
    R = (e.β / length(c_new))^2 * abs(eval(c_new, e) / eval(c_old, e))
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
    R = (length(c_old) / e.β)^2 * abs(eval(c_new, e) / eval(c_old, e))
    return R
end

function finalize(move::RemovalMove, c::Configuration)
    c_new = c + move
    return c_new
end


function metropolis_hastings_update(c, e)

    # select random move
    moves = [ new_insert_move, new_removal_move ]
    move_idx = rand(1:length(moves))

    move = moves[move_idx](c, e)
    
    R = propose(move, c, e)
    
    if R > rand()
        c = finalize(move, c)
    end
    
    return c
end

mutable struct GreensFunction
    β::Float64
    data::Vector{Float64}
    sign::Float64
    function GreensFunction(β::Float64, data::Vector{Float64}, sign::Float64)
        new(β, data, sign)
    end
    function GreensFunction(β::Float64, data::Vector{Float64})
        new(β, data, 0.0)
    end
    function GreensFunction(β::Float64, N::Int64)
        new(β, zeros(N), 0.0)
    end
end

Base.:length(g::GreensFunction) = length(g.data)
Base.:(*)(g::GreensFunction, scalar::Float64) = GreensFunction(g.β, g.data * scalar, g.sign)
Base.:(/)(g::GreensFunction, scalar::Float64) = GreensFunction(g.β, g.data / scalar, g.sign)

function accumulate!(g::GreensFunction, time::Float64, value::Float64)
    if time < 0.0
        value *= -1
        time += g.β
    end
    #@assert time >= 0.0
    #@assert time <= g.β
    idx = ceil(Int, length(g) * time / g.β)
    g.data[idx] += value
end

function sample_greens_function!(g::GreensFunction, c::Configuration, e::Expansion)

    d = Determinant(c, e)
    M = LinearAlgebra.inv(d.mat)

    w = trace(c, e) * d.value
    g.sign += sign(w)
    
    nt = length(c)
    for i in 1:nt, j in 1:nt
        accumulate!(g, c.t_f[i] - c.t_i[j], M[j, i])
    end
end

function semi_circular_g_tau(times, t, h, β)

    np = PyCall.pyimport("numpy")
    kernel = PyCall.pyimport("pydlr").kernel
    quad = PyCall.pyimport("scipy.integrate").quad

    #def eval_semi_circ_tau(tau, beta, h, t):
    #    I = lambda x : -2 / np.pi / t**2 * kernel(np.array([tau])/beta, beta*np.array([x]))[0,0]
    #    g, res = quad(I, -t+h, t+h, weight='alg', wvar=(0.5, 0.5))
    #    return g

    g_out = zero(times)
    
    for (i, tau) in enumerate(times)
        I = x -> -2 / np.pi / t^2 * kernel([tau/β], [β*x])[1, 1]
        g, res = quad(I, -t+h, t+h, weight="alg", wvar=(0.5, 0.5))
        g_out[i] = g
    end

    return g_out
end

β = 20.0
h = 0.0
t = 1.0

nt = 200
times = range(0., β, nt)
g_ref = semi_circular_g_tau(times, t, h, β)

Δ = Hybridization(times, -0.25 * t^2 * g_ref, β)

e = Expansion(β, h, Δ)

println("Starting CT-HYB QMC")

epoch_steps = 10
warmup_epochs = 1000
sampling_epochs = Int(1e5)

g = GreensFunction(β, nt)
c = Configuration([], [])

println("Warmup epochs $warmup_epochs with $epoch_steps steps.")

for epoch in 1:warmup_epochs
    for step in 1:epoch_steps
        global c
        c = metropolis_hastings_update(c, e)
    end    
end

println("Sampling epochs $sampling_epochs with $epoch_steps steps.")

for epoch in 1:sampling_epochs
    for step in 1:epoch_steps
        global c
        c = metropolis_hastings_update(c, e)
    end
    sample_greens_function!(g, c, e) 
end

@show g.sign

dt = β/length(g)
g /= -g.sign * β * dt

# Plot gf, cf exact result!
import PyPlot as plt

dt = β/nt
t = range(0.5*dt, β-0.5*dt, nt)
plt.plot(t, g.data, ".", label="G")
plt.plot(times, g_ref, "-", label="G (ref)")

times = collect(range(0, β, 1001))
plt.plot(times, Δ(times), "-", label="Delta")
plt.plot(Δ.times, Δ.values, ".", label="Delta")

plt.legend(loc="best")
#plt.ylim([-1, 0])
plt.show()
