# Chapter 2, problem 6

h = [0.2, 0.1, 0.05]
b = 1:5
temp = []
for i in h
    for j in 1:5
        err = 0.5*i*(exp(j) - 1)
        push!(temp,[i, j, err])
    end
end
bounds = reduce(hcat,temp)'

# Chapter 2, problem 7

h = [0.2, 0.1, 0.05]
b = 1:5
temp = []
for i in h
    for j in 1:5
        err = 0.5*i*j
        push!(temp,[i, j, err])
    end
end
bounds = reduce(hcat,temp)'

# Chapter 2, problem 8

h = [0.2, 0.1, 0.05]
b = 1:6
temp = []
for i in h
    for j in 1:6
        err = 0.5*i*(exp(j) - 1)
        push!(temp,[i, j, err])
    end
end
bounds = reduce(hcat,temp)'

# Chapter 2, problem 14

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

# differential equation
function func(t, y)
    dy = -y + 2*cos(t)
end

# solve
y0 = 1.0
t0 = 0.0
T = [1,2,3,4,5,6]
n_1 = 20 .* T
n_2 = 10 .* T

temp = []
for i in T
    sol_1 = euler(func, t0, T[i], y0, n_1[i])
    sol_2 = euler(func, t0, T[i], y0, n_2[i])
    y_ext = 2*last(sol_1[2]) - last(sol_2[2])
    push!(temp, y_ext)
end

println(temp)
#println(sol1)

#n_2 = 3

#sol2 = euler(func, t0, T, y0, n_2)

#y_extrapolate = 2 .* sol1[2] - sol2[2]

#print(y_extrapolate)

