using BenchmarkTools

function rk4(f, h::Float64, n::Int, t0::Float64, y0::Vector{Float64}, var)
    t = t0
    y = y0
    for i = 1:n
        s1 = h .* f(t, y, var)
        s2 = h .* f((t + 0.5*h), (y + 0.5 .* s1), var)
        s3 = h .* f((t + 0.5*h), (y + 0.5 .* s2), var)
        s4 = h .* f(t + h, (y .+ s3), var)

        y = y .+ (s1 + 2*s2 + 2*s3 + s4) ./ 6
        t = t + h
    end
    return y
end

function test_func(t::Float64, y::Vector{Float64}, var)
    U = [y[2], y[3], 3*y[1] - y[4] + t^2, 2*y[1] + 4*y[4] + t^3 + 1]
    return U
end

# benchmark
# @btime rk4(test_func, 0.001, 1000, 0.0, [0.2, 3.1, -1.2, 0.0], 0)
# 288 microseconds