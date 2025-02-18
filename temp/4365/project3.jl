using OrdinaryDiffEq, Plots, LaTeXStrings, Plots.PlotMeasures, LinearAlgebra

# Exercise 1

E = 10
L = 0.1 
C = 0.1
R = [1, 10, 100, 1000, 10000]

t = range(0, 4, length = 10000)
all_plots = []

for i in eachindex(R)
    # linear system
    A = [0 1; -1/(C*L) -R[i]/L]
    f = [0; E/L]
    q_0 = [0; 0]

    # particular solution
    q_p = - A \ f

    # homogenous initial value
    q_h0 = q_0 - q_p

    # solution
    q = []
    for j in eachindex(t)
        push!(q,exp(t[j].*A)*q_h0 + q_p)
    end

    local sol = mapreduce(permutedims, vcat, q)

    # plot
    local p = plot(t, [sol[:,1], sol[:,2]],
        xlabel = L"t",
        ylabel = L"q_j(t)", 
        title = string(L"R=", latexstring(R[i])), 
        label = [L"q(t)" L"q'(t)"], 
        linewidth = 2,
        legend = :topright
    )

    push!(all_plots, p)
end

figure1 = plot(all_plots[1], 
    all_plots[2],
    all_plots[3],
    all_plots[4],
    all_plots[5],
    layout = (3, 2),
    size=(800,1200),
    margin = 2mm
)

savefig(figure1, "C:\\Users\\miles\\Desktop\\numerical differential equations\\projects\\project 3\\figure1.pdf")


# Exercise 2
# part a

K = [0, 1, 4, 8]
t = range(0, 10, length = 100000)

all_plots = []

for i in eachindex(K)
    # linear system
    A = [0 1 0; 0 0 1; -K[i] -2 -3]
    y_0 = [1; 1; 1]

    # solution
    y = []
    for j in eachindex(t)
        push!(y,exp(t[j].*A)*y_0)
    end

    local sol = mapreduce(permutedims, vcat, y)

    # plot
    local p = plot(t, [sol[:,1], sol[:,2], sol[:,3]],
        xlabel = L"t",
        ylabel = L"y_j(t)", 
        title = string(L"K=", latexstring(K[i])), 
        label = [L"y(t)" L"y'(t)" L"y''(t)"], 
        linewidth = 2,
    )

    push!(all_plots, p)
end

figure2a = plot(all_plots[1], 
    all_plots[2],
    all_plots[3],
    all_plots[4],
    layout = (2, 2),
    size=(800,800)
)
savefig(figure2a, "C:\\Users\\miles\\Desktop\\numerical differential equations\\projects\\project 3\\figure2a.pdf")


# part b

# function
function func2b(U)
    return [
        5.0*U[1] + 4.0*U[2] - U[1]*U[3],
        U[1] + 4.0*U[2] - U[2]*U[3],
        U[1]^2 + U[2]^2 - 89
    ]
end

# Jacobian
function dfunc2b(U)
    return [
        [5.0 - U[3] 4.0 -1.0*U[1]];
        [1.0 4.0 - U[3] -1.0*U[2]];
        [2.0*U[1] 2.0*U[2] 0]
    ]
end

# Newton's method
function newton(g, dg, U0, tol, nmax)
    # U0: initial guess
    # tol: error allowed
    # nmax: maximum iterations
    U = U0
    for i in 1:nmax
        Unew = U - (dg(U) \ g(U))
        if norm(Unew - U, 2) < tol
            return Unew
        end
        U = Unew
    end
    return U
end

# initial guesses
points = [
    [8, -5, 2],
    [-8, 5, 2],
    [9, 3, 7],
    [-9, -3, 7]]

for i in eachindex(points)
    x = newton(func2b, dfunc2b, points[i], 10^(-15), 20)
    println(x)
end


# Exercise 3

# part a

# differential equation
function brusselatorode(dy, y, p, t)
    a = p[1]
    b = p[2]

    dy[1] = a - (b + 1)*y[1] + (y[1])^2 * y[2]
    dy[2] = b*y[1] - (y[1])^2 * y[2]
end

# part b

# initial conditions
T = 30.0
tspan = [0, T]
y0 = [1.0, 1.0]
p = [1, 3]

# solve
prob = ODEProblem(brusselatorode, y0, tspan, p)
sol = solve(prob, BS3())

# plot
plot3b_1 = plot(sol, idxs=[1,2],
    xaxis = L"t",
    yaxis = "concentration",
    title = "Concentration",
    label = [L"U(t)" L"V(t)"], 
    linewidth = 2,
)

plot3b_2 = plot(sol, idxs=(1,2),
    xaxis = L"U(t)",
    yaxis = L"V(t)",
    title = "Phase plane",
    linewidth = 2,
    legend = false
)

figure3b = plot(plot3b_1, 
    plot3b_2, 
    layout = (1,2),
    size = (800,400),
    margin = 2mm
)

savefig(figure3b, "C:\\Users\\miles\\Desktop\\numerical differential equations\\projects\\project 3\\figure3b.pdf")

# part c

# initial conditions
T = 100.0
tspan = [0, T]
y0 = [1.0, 1.0]
p = [2, 20]

# solve
prob = ODEProblem(brusselatorode, y0, tspan, p)
sol1 = solve(prob, BS3())
sol2 = solve(prob, Rosenbrock23())

# plot
plot3c_1 = plot(sol1, idxs=[1,2],
    xaxis = L"t",
    yaxis = "concentration",
    title = "Concentration, ode23",
    label = [L"U(t)" L"V(t)"], 
    linewidth = 2
)

plot3c_2 = plot(sol1, idxs=(1,2),
    xaxis = L"U(t)",
    yaxis = L"V(t)",
    title = "Phase plane, ode23",
    linewidth = 2,
    legend = false
)

plot3c_3 = plot(sol2, idxs=[1,2],
    xaxis = L"t",
    yaxis = "concentration",
    title = "Concentration, ode23s",
    label = [L"U(t)" L"V(t)"], 
    linewidth = 2,
)

plot3c_4 = plot(sol2, idxs=(1,2),
    xaxis = L"U(t)",
    yaxis = L"V(t)",
    title = "Phase plane, ode23s",
    linewidth = 2,
    legend = false
)

figure3c = plot(plot3c_1,
    plot3c_2,
    plot3c_3,
    plot3c_4,
    layout = (2,2),
    size=(800,800)
)

savefig(figure3c, "C:\\Users\\miles\\Desktop\\numerical differential equations\\projects\\project 3\\figure3c.pdf")

# Exercise 4
# part b

# Adams-Bashforth 2-step
function ab2(z)
    x = (2 .* (z.^2 - z)) ./ (3 .* z - 1)
    return x
end

# Adams-Moulton 2-step

function am2(z)
    x = (12 .* (z.^2 - z)) ./ (5 .* z.^2 + 8 .* z - 1)
    return x
end

# visualization
z = exp.(im.*range(0, 2*pi, length = 10000))
figure4b = plot(ab2.(z),
    title = "Regions of stability", 
    label = "AB-2", 
    linewidth = 2,
    size=(400,400)
)
plot!(figure4b, am2.(z),
    xlabel = L"\mathrm{Re}(z)",
    ylabel = L"\mathrm{Im}(z)", 
    label = "AM-2",
    linewidth = 2,
)

savefig(figure4b, "C:\\Users\\miles\\Desktop\\numerical differential equations\\projects\\project 3\\figure4b.pdf")



# Exercise 5
# part a

function shooting_function(x, y, var)
    p = var[1]
    q = var[2]
    f = var[3]

    return [y[2], -q*y[1] - p*x*y[2] + f]
end

# part b

function RK4class(f, t0::Float64, T::Float64, y0, n, var)
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

function shooting(func, a, b, b_c, n, var)
    alpha = b_c[:,1]
    beta = b_c[:,2]
    eta = b_c[:,3]

    # solve for c
    if alpha[2] == 0
        c = [1, -1/alpha[1]]
    else 
        c = [(-1-alpha[1])/alpha[2], 1]
    end

    # homogenous solution
    yh_0 = [-alpha[2], alpha[1]]
    y_h = RK4class(func, a, b, yh_0, n, [var[1], var[2], 0])

    # particular solution
    yp_0 = -eta[1] .* [c[2], c[1]]
    y_p = RK4class(func, a, b, yp_0, n, var)

    # solving for s
    yh_b = y_h[2][:, n+1]
    yp_b = y_p[2][:, n+1]

    s = (eta[2] - dot(beta, yp_b))/dot(beta,yh_b)

    y = [y_h[1], y_p[2] .+ s .* y_h[2]]

    return y
end

# part c

# parameters
p_5 = 3.0
q_5 = -2.0
f_5 = -5.0
var_5 = [p_5, q_5, f_5]

# interval
a = 0.0
b = 5.0

# time steps
n = 10000

# boundary conditions
BC = [[0.0 1.0 1.0]; [1.0 0.0 -1.0]]

# solution
sol = shooting(shooting_function, a, b, BC, n, var_5)

# plot
figure5 = plot(sol[1], sol[2][1,:],
    title = "Shooting method",
    xlabel = L"x",
    ylabel = L"y", 
    label = L"y(x)",
    linewidth = 2
)
plot!(figure5, sol[1], sol[2][2,:],
    label = L"y'(x)",
    linewidth = 2
)

savefig(figure5, "C:\\Users\\miles\\Desktop\\numerical differential equations\\projects\\project 3\\figure5.pdf")