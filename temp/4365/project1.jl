using OrdinaryDiffEq, Plots, LaTeXStrings, Plots.PlotMeasures

# Exercise 1

# function y(t) = exp(0.5*t)
function y(t)
    return exp.(0.5.*t)
end

# derivative y'(t) = 0.5*exp(0.5*t)
function dy(t)
    return 0.5.*exp.(0.5.*t)
end

# forward difference approximation
function forward_difference(y, t, h)
    return (y(t .+ h) .- y(t))./h
end

# centered difference approximation
function centered_difference(y,t,h)
    return (y(t .+ h) .- y(t .- h))./(2 .* h)
end
  
# approximation error
function error_h(y, dy, t, h, approximation)
    return abs.(dy(t) .- approximation(y,t,h))
end

# solve
t = 1
h = 10 .^ range(-14, stop = -1, length = 200)
error_forward = error_h(y, dy, t, h, forward_difference)
error_centered = error_h(y, dy, t, h, centered_difference)

# plot
figure1 = plot(h,[error_forward,error_centered], 
    xaxis = :log,
    yaxis = :log,
    xlabel = L"\log_{10} (h)",
    ylabel = L"\log_{10} (\textrm{error})",
    xlims = (10^(-15), 0),
    title = "Error of approximations",
    label = ["forward difference" "centered difference"], 
    linewidth = 2,
    legend = :bottomleft
)

savefig(figure1, "C:\\Users\\miles\\Desktop\\numerical differential equations\\projects\\project 1\\figure1.pdf")


# Exercise 2

# part a
function euler(f, t0, T, y0, n)
    y = y0
    t = t0
    h = (T - t0)/n

    t_sol = Float64[t0]
    y_sol = Float64[y0]
    
    for i in 1:n
        y = y + h*f(t, y)
        t = t + h

        push!(t_sol, t)
        push!(y_sol, y)
    end

    return [t_sol,y_sol]
end

# part b

# differential equation dy = 2*t*y^2
function func_2b(t, y)
    dy = 2*t*y^2
end

# solve
y0 = 1.0
t0 = 0.0
T = 0.5
n = 100

sol = euler(func_2b, t0, T, y0, n)

figure2b = plot(sol[1],sol[2], 
    xlabel = L"t",
    ylabel = L"y(t)",
    title = string("Euler's method for ", L"y'(t) = 2ty^2"),
    linewidth = 2,
    legend = false
)

savefig(figure2b, "C:\\Users\\miles\\Desktop\\numerical differential equations\\projects\\project 1\\figure2b.pdf")

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

# part d

# differential equation
function pendulumode(t, y, var)
    g = var[1]
    l = var[2]

    dy = Float64[0,0]
    dy[1] = y[2]
    dy[2] = (-g/l) * sin(y[1])

    return dy
end

# part e

# solve
g = 9.81
l = 0.1
var = [g, l]

n = 200
t0 = 0.0
T = 2.0
y0 = Float64[0.05, 0]

sol = eulersys(pendulumode, t0, T, y0, n, var)

# plot
figure2e = plot(sol[1], sol[2][1,:], 
    xlabel = L"t",
    ylabel = L"y_j(t)",
    title = "Pendulum",
    label = L"y_1(t)", 
    linewidth = 2,
)

plot!(figure2e, sol[1], sol[2][2,:],
    label = L"y_2(t)",
    linewidth = 2
)

savefig(figure2e, "C:\\Users\\miles\\Desktop\\numerical differential equations\\projects\\project 1\\figure2e.pdf")


# Exercise 3

# differential equation
function satellite(dy, y, p, t)
    mu = 1/82.45
    mu_p = 1-mu

    dy[1] = y[3]
    dy[2] = y[4]
    dy[3] = y[1] + 2*y[4] - mu_p * (y[1]+mu) / (((y[1]+mu)^2 + y[2]^2)^(3/2)) - mu * (y[1]-mu_p) / (((y[1]-mu_p)^2 + y[2]^2)^(3/2))
    dy[4] = y[2] - 2*y[3] - mu_p * y[2] / (((y[1]+mu)^2 + y[2]^2)^(3/2)) - mu * y[2] / (((y[1]-mu_p)^2 + y[2]^2)^(3/2))
end

# initial conditions
T = 6.19
tspan = [0, T]
y0 = [1.2, 0, 0, -1.05]


# solve
prob = ODEProblem(satellite, y0, tspan)
sol = solve(prob, DP5(), abstol=1e-8, reltol=1e-7)

# plot
figure3a = plot(sol, idxs=(1,2),
    xaxis = L"x(t)",
    yaxis = L"y(t)",
    title = "Satellite position",
    linewidth = 2,
    legend = false
)

scatter!(figure3a, [0], [0],
    color = "green",
    markersize = 6)

scatter!(figure3a, [1], [0],
    color = "gray",
    markersize = 4)

scatter!(figure3a, [1.2], [0],
    color = "black")

quiver!(figure3a, [1.2], [0],
    quiver = ([0], [-0.2],),
    color = "black")

annotate!([0, 1, 1.3], [0.07, 0.07, 0], 
    [Plots.text("Earth", 9), 
    Plots.text("moon", 9), 
    Plots.text(L"t = 0",8)])

figure3b = plot(sol, idxs=[3,4],
    xaxis = L"t",
    yaxis = "velocity",
    title = "Satellite velocity",
    label = [string(L"x", "-velocity") string(L"y", "-velocity")], 
    linewidth = 2
)

savefig(figure3a, "C:\\Users\\miles\\Desktop\\numerical differential equations\\projects\\project 1\\figure3a.pdf")
savefig(figure3b, "C:\\Users\\miles\\Desktop\\numerical differential equations\\projects\\project 1\\figure3b.pdf")


# Exercise 4

# differential equation
function predatorprey(dy, y, p, t)
    dy[1] = (a - e)*y[1] - b*y[2]*y[1]
    dy[2] = -(c + e)*y[2] + d*y[1]*y[2]
end

# initial conditions
y0 = [20, 20]
tspan = (0.0, 25.0)

# parameters no fishing
param1 = [1,0.03,1.2,0.02,0]
a = param1[1]
b = param1[2]
c = param1[3]
d = param1[4]
e = param1[5]

# solve
prob1 = ODEProblem(predatorprey, y0, tspan)
sol1 = solve(prob1, DP5())

# parameters with fishing
param2 = [1,0.03,1.2,0.02,0.4]
a = param2[1]
b = param2[2]
c = param2[3]
d = param2[4]
e = param2[5]

# solve
prob2 = ODEProblem(predatorprey, y0, tspan)
sol2 = solve(prob2, DP5())

# plot
figure4a = plot(sol1, 
    yaxis = "population", 
    title = "Predator-prey model", 
    label = ["prey" "predator"], 
    linewidth = 2,
    legend = :topright
)
plot!(figure4a, sol2,
    xaxis = L"t", 
    label = ["prey with fishing" "predator with fishing"], 
    linewidth = 2
)

figure4b = plot(sol1, idxs = (1,2),
    xaxis = "prey",
    yaxis = "predator",
    title = "Predator-prey model, phase plane plot",
    label = "without fishing",
    linewidth = 2
)
plot!(figure4b, sol2, idxs = (1,2),
    xlims = (0,250),
    label = "with fishing",
    linewidth = 2,
    right_margin = 2mm
)

savefig(figure4a, "C:\\Users\\miles\\Desktop\\numerical differential equations\\projects\\project 1\\figure4a.pdf")
savefig(figure4b, "C:\\Users\\miles\\Desktop\\numerical differential equations\\projects\\project 1\\figure4b.pdf")


# Exercise 5

# differential equation
function chemistry(dy,y,p,t)
    dy[1] = -k1*y[1] + k2*y[2]*y[3]
    dy[2] = k1*y[1] - k2*y[2]*y[3] - k3*y[2]*y[2]
    dy[3] = k3*y[2]*y[2]
end

# initial conditions
y0 = [1, 0, 0]
tspan = (0.0, 1000.0)

# parameters
k1 = 0.04
k2 = 1e4
k3 = 3e7

# solve
prob = ODEProblem(chemistry, y0, tspan)
sol = solve(prob, TRBDF2())

# plot
figure5 = plot(sol, 
    xaxis = L"t",
    yaxis = L"y_j (t)",
    title = "Autocatalytic chemical reaction",
    label = [L"y_1(t)" L"y_2(t)" L"y_3(t)"],
    linewidth = 2,
    right_margin = 3mm
)

savefig(figure5, "C:\\Users\\miles\\Desktop\\numerical differential equations\\projects\\project 1\\figure5.pdf")