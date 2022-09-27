
using LinearAlgebra

struct Configuration
    t_i::Vector{Float64}
    t_f::Vector{Float64}
end

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

struct Determinant
    mat::Array{Float64, 2}
    value::Float64

    function Determinant(c::Configuration, Δ::Hybridization)
        mat = [ Δ(tf - ti) for tf in c.t_f, ti in c.t_i ]
        value = LinearAlgebra.det(mat)
        new(mat, value)
    end
end

function trace(c::Configuration)
end

struct InsertMove
    t_i::Float64
    t_f::Float64
end

struct RemovalMove
    i_idx::UInt64
    f_idx::UInt64
end



function propose(move::InsertMove, c::Configuration)
end

function finalize(move::InsertMove, c::Configuration)
end





N = 3
t_i = rand(Float64, N)
t_f = rand(Float64, N)

@show t_i
@show t_f

config = Configuration(t_i, t_f)

@show config

times = range(0, 1, 4)

ϵ = 1.
values = exp.(-ϵ .* times)
@show times
@show values

Δ = Hybridization(times, values)

@show Δ

det = Determinant(config, Δ)

@show det
