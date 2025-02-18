using LinearAlgebra, Plots, LaTeXStrings

# Exercise 5
# part a
println(127/13+pi)
println(exp(3)*13*im)
println(exp(pi*im))

# part b
C = [0.5; 2; 1]
B = [2 3 2; -0.25 0 -1; 3/5 -3/5 0]
println(C*transpose(C))
println(transpose(C) * B * C)

# part c
println(23+10^37+1000-10^37)


# Exercise 6
# part a
println(norm(C))
println(sum(C)/length(C))

# part b 
println(det(B))
println(inv(B))

# part c
println(eigvals(B))
println(eigvecs(B))


# Exercise 9
# part a
function f_stk(x)
    n = length(x)
    f = Float64[]
    
    for i in 1:n
        if x[i] < 0
            push!(f, -sin(x[i]))
        elseif 0 <= x[i] && x[i] <= 1
            push!(f, (x[i])^2)
        else x[i] > 1
            push!(f, 1/x[i])
        end
    end

    return f
end

# part b
x = range(-pi/2, pi, length = 1200)
y = reduce(vcat,[f_stk(i) for i in x])

figure9b = plot(x,y,
    xlabel = L"x",
    ylabel = L"f(x)",
    linewidth = 2,
    legend = false)

savefig(figure9b, "C:\\Users\\miles\\Desktop\\numerical differential equations\\projects\\project 2\\figure9b.pdf")


# Exercise 12

# part a
function impr_euler(f, t0, T, y0, n, var)
    y = y0
    t = t0
    h = (T - t0)/n

    t_sol = [t0]
    y_sol = [y0]
    
    for i in 1:n
        s_1 = f(t, y, var)
        s_2 = f(t + h, y .+ h.*s_1,var)
        y = y + 0.5.* h .* (s_1 .+ s_2)
        t = t + h

        push!(t_sol, t)
        push!(y_sol, y)
    end

    y_ssol = reduce(hcat,y_sol)

    return [t_sol,y_ssol]
end

# part b
function pendulumode(t, y, var)
    g = var[1]
    l = var[2]

    dy = Float64[0,0]
    dy[1] = y[2]
    dy[2] = (-g/l) * y[1]

    return dy
end

# part c
function eulersys(f, t0, T, y0, n, var)
    y = y0
    t = t0
    h = (T - t0)/n

    t_sol = [t0]
    y_sol = [y0]
    
    for i in 1:n
        y = y .+ h.*f(t, y, var)
        t = t + h

        push!(t_sol, t)
        push!(y_sol, y)
    end

    y_ssol = reduce(hcat,y_sol)

    return [t_sol,y_ssol]
end

# create approximation errors
function error_n(Y, method, f, t0, T, y0, n, var)
    return norm(Y .- method(f, t0, T, y0, n, var)[2][:,end])
end

# parameters
g = 9.81
l = 0.1
var = [g, l]

t0 = 0.0
T = 2*pi*sqrt(l/g)
x0 = Float64[0.05, 0.0]

parameters = [pendulumode,t0,T,x0,var]
Y = Float64[0.05, 0.0] # exact solution
N = 3

function error_collection(methods, exact, N, params)
    f = params[1]
    t0 = params[2]
    T = params[3]
    x0 = params[4]
    var = params[5]

    errors = []
    h = Float64[]

    for method_i in methods
        method_error_n = Float64[]
        for i in 2:(1+N)
            n = 10^i
            error_of_method = error_n(exact, method_i, f, t0, T, x0, n, var)
            push!(method_error_n, error_of_method)
        end
        push!(errors, method_error_n)
    end
    for j in 2:(1+N)
        n = 10^j
        push!(h, 1/n)
    end

    return [h, errors]
end

E = error_collection([eulersys,impr_euler], Y, N, parameters)
figure12c = plot(E[1],[E[2][1],E[2][2]],
    xaxis = :log,
    yaxis = :log,
    xlabel = L"\log_{10} (h)",
    ylabel = L"\log_{10} (\textrm{error})",
    title = "Error of approximations",
    label = ["explicit Euler" "improved Euler"], 
    linewidth = 2,
    legend = :bottomright
)
savefig(figure12c, "C:\\Users\\miles\\Desktop\\numerical differential equations\\projects\\project 2\\figure12c.pdf")


# Exercise 13

# part b
function fixed_point(phi,x0,tol,n_max)
    x = x0
    i = 1

    while i <= n_max
        x_i = x
        x = phi(x_i)
        if abs(x - x_i) < tol
            return [x,i]
        end
        i = i + 1
    end
    return [x,n_max]
end

# part c
function phi_1(x)
    return x - x^3 - 4*x^2 + 10
end

function phi_2(x)
    return 0.5*(10 - x^3)^(0.5)
end

function phi_3(x)
    return x - ((x^3 + 4*x^2 -10)/(3*x^2 + 8*x))
end

results = []
for phi_i in [phi_1, phi_2, phi_3]
    push!(results, fixed_point(phi_i, 1.5, 10^(-5), 15))
end

println(results)

# graphing
x = range(1,2, length=1000)
y_1 = phi_1.(x)
y_2 = phi_2.(x)
y_3 = phi_3.(x)

figure13c = plot(x, [y_1,y_2,y_3],
    xlabel = L"x",
    label = [L"\phi_1(t)" L"\phi_2(t)" L"\phi_3(t)"],
    linewidth = 2,
    ylims = (1,2)
)

savefig(figure13c, "C:\\Users\\miles\\Desktop\\numerical differential equations\\projects\\project 2\\figure13c.pdf")


# Exercise 14

# part a
function RK4class(f, t0, T, y0, n, var)
    y = y0
    t = t0
    h = (T - t0)/n

    t_sol = [t0]
    y_sol = [y0]
    
    for i in 1:n

        s_1 = f(t, y, var)
        s_2 = f(t + 0.5*h, y .+ 0.5 .* h .* s_1, var)
        s_3 = f(t + 0.5*h, y .+ 0.5 .* h .* s_2, var)
        s_4 = f(t + h, y .+ h.*s_3, var)

        y = y .+ (1/6) .* h.* (s_1 .+ 2 .* s_2 .+ 2 .* s_3 .+ s_4)
        t = t + h

        push!(t_sol, t)
        push!(y_sol, y)
    end

    y_ssol = reduce(hcat,y_sol)

    return [t_sol,y_ssol]
end

# part b

# parameters
g = 9.81
l = 0.1
var = [g, l]

t0 = 0.0
T = 2*pi*sqrt(l/g)
x0 = Float64[0.05, 0.0]

parameters = [pendulumode,t0,T,x0,var]
Y = Float64[0.05, 0.0] # exact solution
N = 3

E = error_collection([eulersys,impr_euler,RK4class], Y, N, parameters)
figure14b = plot(E[1],[E[2][1],E[2][2],E[2][3]],
    xaxis = :log,
    yaxis = :log,
    xlabel = L"\log_{10} (h)",
    ylabel = L"\log_{10} (\textrm{error})",
    title = "Error of approximations",
    label = ["explicit Euler" "improved Euler" "classical Runge-Kutta"], 
    linewidth = 2,
    legend = :bottomright
)
savefig(figure14b, "C:\\Users\\miles\\Desktop\\numerical differential equations\\projects\\project 2\\figure14b.pdf")

# part c
function func14c(t, y, var)
    dy = t^0.5
    return dy
end

# parameters
t0 = 0.0
T = 1
y0 = 0.0
var = []

parameters = [func14c,t0,T,y0,var]
Y = 1.5 # exact solution
N = 3

E = error_collection([eulersys,impr_euler,RK4class], Y, N, parameters)
figure14c = plot(E[1],[E[2][1],E[2][2],E[2][3]],
    xaxis = :log,
    yaxis = :log,
    xlabel = L"\log_{10} (h)",
    ylabel = L"\log_{10} (\textrm{error})",
    title = "Error of approximations",
    label = ["explicit Euler" "improved Euler" "classical Runge-Kutta"], 
    linewidth = 2,
    legend = :topleft
)
savefig(figure14c, "C:\\Users\\miles\\Desktop\\numerical differential equations\\projects\\project 2\\figure14c.pdf")