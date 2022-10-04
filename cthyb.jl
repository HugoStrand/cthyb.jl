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
        #return length(t_i) == 0 ? new([], []) : new(sort(t_i), sort(t_f))
        #return length(t_i) == 0 ? new([], []) : new(t_i, t_f)
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
    #@show time
    #s = time < 0. ? -1.0 : 1.0
    #@show s
    #time = mod(time, Δ.β)
    s = 1.0
    if time < 0.0
        s = -1.0
        time += Δ.β
    end
    #@show time
    idx = searchsortedfirst(Δ.times, time)
    #@show idx
    if idx == 1
        idx = 2
    end
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
        #if length(c) == 0
        #    return new(Matrix{Float64}(undef, 0, 0), 1.0)
        #end
        
        if true
            if length(c) > 0 && first(c.t_f) < first(c.t_i)
                t_f = copy(c.t_f)
                push!(t_f, popfirst!(t_f))
                mat = [ e.Δ(tf - ti) for tf in t_f, ti in c.t_i ]
            else
                mat = [ e.Δ(tf - ti) for tf in c.t_f, ti in c.t_i ]
            end
        else
            mat = [ e.Δ(tf - ti) for tf in c.t_f, ti in c.t_i ]
        end
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
    #@show ops
    
    first_state = first(ops).operation > 0 ? 0 : 1
    
    state = copy(first_state)
    t_i = 0.0

    #value = (first_state == 1) ? (-1.0)^(length(c)-1) : +1.0
    value = (first_state == 1) ? -1.0 : +1.0
    #value = (first_state == 1) ? +1.0 : -1.0
    #value = 1.0

    #@show value
    #@show first_state
    
    for op in ops
        t_f = op.time

        if state == 1
            dt = t_f - t_i
            value *= exp(-e.h * dt)
        end
        
        t_i = t_f
        state += op.operation

        if state < 0 || state > 1
            return 0.0
        end

        #@show value
        #@show state
    end

    t_f = e.β

    if state == 1
        dt = t_f - t_i
        value *= exp(-e.h * dt)
    end
    
    #@show value
    if first_state != state
        return 0.0
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

    # select move
    moves = [ new_insert_move, new_removal_move ]
    move_idx = rand(1:length(moves))
    move = moves[move_idx](c, e)
    #@show move
    
    R = propose(move, c, e)
    #@show R
    
    # accept/reject move
    if R > rand()
        c = finalize(move, c)
    end
    #@show length(c)
    
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
    @assert time >= 0.0
    @assert time <= g.β
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

N_t = 100

times = range(0., β, N_t)

#ϵ = 0.0
#V = 1.0
#values = -V^2 * exp.(-ϵ .* times) / (1 + exp(-ϵ * β))
#Δ = Hybridization(times, values)
#@show Δ

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

g_ref = semi_circular_g_tau(times, t, h, β)
#@show g_ref

Δ = Hybridization(times, -0.25 * t^2 * g_ref, β)
#exit()

e = Expansion(β, h, Δ)
@show e.β
@show e.h

println("Starting CT-HYB QMC")

chunk = 10
warmup = 1000
sampling = 100000

nt = 200

c = Configuration([], [])
@show c
w = eval(c, e)
@show w

c = Configuration([1.0], [2.0])
@show c
d = Determinant(c, e)
@show d.mat
@show d.value
w = eval(c, e)
@show w

c = Configuration([2.0], [1.0])
@show c
d = Determinant(c, e)
@show d.mat
@show d.value
@show trace(c, e)
w = eval(c, e)
@show w

#exit()

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
