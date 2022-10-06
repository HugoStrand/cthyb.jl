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

        return new(sort(t_i), sort(t_f))
    end
end

Base.:length(c::Configuration) = length(c.t_i)

# -- Segment utilities

struct Segment
    t_i::Float64
    t_f::Float64
end

Base.length(s::Segment, β::Float64) = s.t_i < s.t_f ? s.t_f - s.t_i : β - s.t_i + s.t_f

struct SegmentIterator
    c::Configuration
end

Base.eltype(::Type{SegmentIterator}) = Segment
Base.length(s::SegmentIterator) = length(s.c)

segments(c::Configuration) = SegmentIterator(c::Configuration)

indices(s::SegmentIterator, idx::Int) = s.c.t_f[1] > s.c.t_i[1] ? (idx, idx) : (idx, mod(idx + 1, 1:length(s.c)))

function Base.iterate(s::SegmentIterator, state=1)
    c = s.c
    l = length(c)
    i_idx, f_idx = indices(s, state)
    if i_idx <= l
        return (Segment(c.t_i[i_idx], c.t_f[f_idx]), state + 1)
    else
        return nothing
    end
end

function remove_segment!(c::Configuration, segment_idx::Int)
    @assert 0 < segment_idx <= length(c)
    i_idx, f_idx = indices(segments(c), segment_idx)
    deleteat!(c.t_i, i_idx)
    deleteat!(c.t_f, f_idx)
end

# -- Anti-segment utilities

struct AntiSegment
    t_i::Float64
    t_f::Float64
end

struct AntiSegmentIterator
    c::Configuration
end

Base.length(s::AntiSegmentIterator) = length(s.c)

antisegments(c::Configuration) = AntiSegmentIterator(c::Configuration)

indices(s::AntiSegmentIterator, idx::Int) = s.c.t_f[1] < s.c.t_i[1] ? (idx, idx) : (idx, mod(idx + 1, 1:length(s.c)))

function Base.iterate(s::AntiSegmentIterator, state=1)
    c = s.c
    l = length(c)
    f_idx, i_idx = indices(s, state)
    if f_idx <= l
        return (AntiSegment(c.t_f[f_idx], c.t_i[i_idx]), state + 1)
    else
        return nothing
    end
end

function onsegment(t::Float64, s::Segment)
    if s.t_i < s.t_f
        return t > s.t_i && t < s.t_f
    else
        return t < s.t_f || t > s.t_i
    end
end

""" O(N) algorithm. To do: Use the sorted times for O(logN) """
function onsegment(t::Float64, c::Configuration)
    for s in segments(c)
        if onsegment(t, s)
            return s
        end
    end
    return nothing
end

function onantisegment(t::Float64, s::AntiSegment)
    if s.t_i < s.t_f
        return t > s.t_i && t < s.t_f
    else
        return t < s.t_f || t > s.t_i
    end
end

""" O(N) algorithm. To do: Use the sorted times for O(logN) """
function onantisegment(t::Float64, c::Configuration)
    for s in antisegments(c)
        if onantisegment(t, s)
            return s
        end
    end
    return nothing
end

function remove_antisegment!(c::Configuration, segment_idx::Int)
    @assert 0 < segment_idx <= length(c)
    f_idx, i_idx = indices(antisegments(c), segment_idx)
    deleteat!(c.t_i, i_idx)
    deleteat!(c.t_f, f_idx)
end

# -- Hybridization

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
    t_i::Vector{Float64}
    t_f::Vector{Float64}
    
    function Determinant(c::Configuration, e::Expansion)

        #mat = [ e.Δ(tf - ti) for tf in c.t_f, ti in c.t_i ]

        t_f = copy(c.t_f)
        t_i = copy(c.t_i)

        if length(t_f) > 0 && first(t_f) < first(t_i)
            push!(t_f, popfirst!(t_f))
        end
        
        mat = [ e.Δ(tf - ti) for tf in t_f, ti in t_i ]
        
        value = LinearAlgebra.det(mat)
        new(mat, value, t_i, t_f)
    end
end

function is_segment_proper(c::Configuration)
    if c.t_i[1] < c.t_f[1]
        for idx in 2:length(c)
            if !( c.t_f[idx - 1] < c.t_i[idx] ) || !( c.t_i[idx] < c.t_f[idx] )
                return false
            end
        end
    else
        for idx in 2:length(c)
            if ! ( c.t_i[idx - 1] < c.t_f[idx] ) || !( c.t_f[idx] < c.t_i[idx] )
                return false
            end
        end
    end
    return true
end

function trace(c::Configuration, e::Expansion)::Float64

    if length(c) == 0
        return 2.0
    end

    if !is_segment_proper(c)
        return 0.0
    end
    
    value = c.t_f[1] < c.t_i[1] ? -1.0 : +1.0
    
    for s in segments(c)
        value *= exp(-e.h * length(s, e.β))
    end

    return value
end

function eval(c::Configuration, e::Expansion)
    return trace(c, e) * Determinant(c, e).value
end

struct InsertMove
    t_i::Float64
    t_f::Float64
    #l::Float64
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
    
    #if length(c) == 0
    #    t_f = e.β * rand()
    #    l = β
    #else
    #    i_idx = searchsortedfirst(c.t_i, t_i)
    #    exit() # Fixme!
    #end

    ## is t_i in a segment?
    
    #return InsertMove(t_i, t_f, l)
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
    l::Float64
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
        #f_idx = i_idx
        #l = c.t_f[move.f_idx] - mod(c.t_i[move.i_idx], e.β)
        return RemovalMove(i_idx, f_idx, l)
    else
        return RemovalMove(0, 0, 0.0)
    end
end

function propose(move::RemovalMove, c_old::Configuration, e::Expansion)
    c_new = c_old + move
    R = (length(c_old) / e.β)^2 * abs(eval(c_new, e) / eval(c_old, e))
    #R = length(c_old) / e.β / move.l * abs(eval(c_new, e) / eval(c_old, e))
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
        accumulate!(g, d.t_f[i] - d.t_i[j], M[j, i])
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

c = Configuration([1.0], [3.0])
d = Determinant(c, e)
t = trace(c, e)
@show c
@show d.mat
@show d.value
@show t
@show is_segment_proper(c)

c = Configuration([19.0], [1.0])
d = Determinant(c, e)
t = trace(c, e)
@show c
@show d.mat
@show d.value
@show t
@show is_segment_proper(c)


c = Configuration([1.0, 5.0], [3.0, 7.0])
d = Determinant(c, e)
t = trace(c, e)
@show c
@show d.mat
@show d.value
@show t
@show is_segment_proper(c)


c = Configuration([2.0, 19.0], [5.0, 1.0])
d = Determinant(c, e)
t = trace(c, e)
@show c
@show d.mat
@show d.value
@show t
@assert is_segment_proper(c)

#exit()
c = Configuration([1.0, 3.0], [2.0, 4.0])

@show c
@assert is_segment_proper(c)


println("Segments")
for s in segments(c)
    @show s
end

println("Anti-segments")
for s in antisegments(c)
    @show s
end

c = Configuration([1.0, 3.0, 5.0], [0.5, 2.0, 4.0])

@show c
@assert is_segment_proper(c)


println("Segments")
for s in segments(c)
    @show s
end

println("Anti-segments")
for s in antisegments(c)
    @show s
end

@show onsegment(1.5, c)
@show onsegment(3.5, c)
@show onsegment(6.0, c)
@show onsegment(0.2, c)
@show onsegment(0.6, c)
@show onsegment(2.5, c)
@show onsegment(4.5, c)

@show onantisegment(1.5, c)
@show onantisegment(3.5, c)
@show onantisegment(6.0, c)
@show onantisegment(0.2, c)
@show onantisegment(0.6, c)
@show onantisegment(2.5, c)
@show onantisegment(4.5, c)

    
@show indices(segments(c), 1)
@show indices(segments(c), 2)
@show indices(segments(c), 3)

for idx in 1:length(c)
    c_tmp = deepcopy(c)
    remove_segment!(c_tmp, idx)
    @show collect(segments(c_tmp))
end

for idx in 1:length(c)
    c_tmp = deepcopy(c)
    remove_antisegment!(c_tmp, idx)
    @show collect(antisegments(c_tmp))
end


c = Configuration([1.0, 2.0], [3.0, 4.0])
@show c
@show is_segment_proper(c)
@assert !is_segment_proper(c)

                
#exit()


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
