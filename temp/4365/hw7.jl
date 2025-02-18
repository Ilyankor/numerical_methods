using LinearAlgebra, SparseArrays, Plots, LaTeXStrings

# Exercise 3

function ex3_diff_func(x)
    return 5*x^3
end

function ex3_exact(x)
    return  0.25*x^5 + 0.75*x - 2
end

function dirichlet(func, initial, final, n)
    # boundary conditions
    a = initial[1]
    b = final[1]
    u_a = initial[2]
    u_b = final[2]

    h = 1/(n-1) # step size
    N = n-2 # number of interior points

    # construct f
    x_i = range(a + h, stop = b - h, length = N)
    y = h^2 .* func.(x_i)

    bc = zeros(N,1)
    bc[1] = u_a
    bc[N] = u_b

    f = y .- bc

    # construct A
    A = Tridiagonal(ones(N-1), -2*ones(N), ones(N-1))
    
    # solve
    u = vec(A \ f)

    # add in boundary points
    x = collect(x_i)

    pushfirst!(x,a)
    pushfirst!(u, u_a)
    push!(x,b)
    push!(u,u_b)
    
    return [x,u]
end

function neumann(func, initial, final, n)
    # boundary conditions
    a = initial[1]
    b = final[1]
    u_a = initial[2]
    du_b = final[2]

    h = 1/(n-1) # step size
    N = n-2 # number of interior points

    # construct f
    x_i = range(a + h, stop = b - h, length = N)
    y = h^2 .* func.(x_i)
    push!(y, 0)

    bc1 = zeros(N+1)
    bc1[1] = u_a

    bc2 = zeros(N+1)
    bc2[N+1] = h*du_b

    f = y .- bc1 .+ bc2

    # construct A
    A = Tridiagonal(ones(N-1), -2*ones(N), ones(N-1))

    side = zeros(N)
    side[N] = 1
    bottom = zeros(N-1)
    append!(bottom, [-1, 1])

    A_s = hcat(A, side)
    A_s_b = vcat(A_s, bottom')

    # solve
    u = vec(A_s_b \ f)

    # add in boundary points
    x = collect(x_i)

    pushfirst!(x,a)
    pushfirst!(u, u_a)
    push!(x,b)
    
    return [x,u]
end

# visualize solutions and approximations
x = range(0, stop = 1, length = 1000000)
u = ex3_exact.(x)

figure3a = plot(x, u,
    title = "Dirichlet boundary",
    label = L"u(x)",
    xlabel = L"x",
    ylabel = L"y",
    linewidth = 2
)
figure3b = plot(x, u,
    title = "Neumann boundary",
    label = L"u(x)",
    xlabel = L"x",
    ylabel = L"y",
    linewidth = 2
)

for i = 4:2:10
    u_approx_d = dirichlet(ex3_diff_func,[0.0, -2.0],[1.0, -1.0], i)
    u_approx_n = neumann(ex3_diff_func,[0.0, -2.0],[1.0, 2.0], i)
    
    plot!(figure3a, u_approx_d[1], u_approx_d[2],
        label = latexstring("i = $(i)"),
        linewidth = 1.5
    )
    plot!(figure3b, u_approx_n[1], u_approx_n[2],
        label = latexstring("i = $(i)"),
        linewidth = 1.5
    )
end

savefig(figure3a, "C:\\Users\\miles\\Desktop\\numerical differential equations\\homework\\hw7\\figure3a.pdf")
savefig(figure3b, "C:\\Users\\miles\\Desktop\\numerical differential equations\\homework\\hw7\\figure3b.pdf")

# calculate error
h_list = Float64[]
error_d_list = Float64[]
error_n_list = Float64[]

for i = 2:13
    j = 2^(i) + 1
    u_approx_d = dirichlet(ex3_diff_func,[0.0, -2.0],[1.0, -1.0], j)
    u_approx_n = neumann(ex3_diff_func,[0.0, -2.0],[1.0, 2.0], j)

    x = range(0, stop = 1, length = j)
    u = ex3_exact.(x)

    h = 1/(j-1)
    
    error_d = maximum(abs.(u .- u_approx_d[2]))
    error_n = maximum(abs.(u .- u_approx_n[2]))

    push!(h_list,h)
    push!(error_d_list, error_d)
    push!(error_n_list, error_n)
end

error_h = [error_d_list ./ (h_list .^ 2), error_n_list ./ h_list]
println(error_h)

# error visualisation
figure3c = plot(h_list, [error_d_list, error_n_list],
    title = string("Error vs. ", L"h"),
    label = ["Dirichlet" "Neumann"],
    xlabel = L"\log_{10} (h)",
    ylabel = L"\log_{10} (\textrm{error})",
    xaxis = :log,
    yaxis = :log,
    legend = :bottomright,
    linewidth = 2
)

savefig(figure3c, "C:\\Users\\miles\\Desktop\\numerical differential equations\\homework\\hw7\\figure3c.pdf")


# Exercise 5

function poisson_square(f, N, I_xy, E)
    # function, uniform number of points, ranges of x and y, boundary functions
    
    # linear system
    # Au = h^2 f - boundary values

    # number of interior points
    n = N - 2 
    h = 1/(N - 1)

    # discrete Laplace operator
    T1 = sparse(Tridiagonal(ones(n-1), -2*ones(n), ones(n-1)))
    I_n = sparse(Diagonal(ones(n)))
    A = kron(T1, I_n) + kron(I_n, T1)

    # ranges
    x_i = I_xy[1][1]
    x_f = I_xy[1][2]

    y_i = I_xy[2][1]
    y_f = I_xy[2][2]

    # grid of x and y
    x = range(x_i, stop = x_f, length = N)
    y = range(y_i, stop = y_f, length = N)

    # boundary functions
    E1 = E[1]
    E2 = E[2]
    E3 = E[3]
    E4 = E[4]

    # boundary values
    e1 = Vector{Float64}([])
    e2 = Vector{Float64}([])
    e3 = Vector{Float64}([])
    e4 = Vector{Float64}([])

    for i = 2:N-1
        push!(e1, E1(x[i], y_i))
        push!(e3, E3(x[i], y_f))
        
        push!(e2, E2(x_i, y[i]))
        push!(e4, E4(x_f, y[i]))
    end

    # basis vectors
    e_1 = zeros(n)
    e_1[1] = 1.0

    e_2 = zeros(n)
    e_2[n] = 1.0

    # sum of boundary values
    sum_edges = kron(e_1, e1) + kron(e2, e_2) + kron(e_2, e3) + kron(e4, e_1)

    # function values
    fval = []
    for i in 1:n
        for j in 1:n
            push!(fval, f(x[i],y[j]))
        end
    end
    
    # solve the system
    b = h^2 .* fval .- sum_edges
    u = A \ b

    # put the whole grid together
    prepend!(u, reduce(vcat,[E1(x_i, y_i), e1, E1(x_f, y_i)]))

    for i in 2:N-1
        insert!(u, (i-1)*N + 1, e4[i-1])
        insert!(u, i*N, e2[i-1])
    end

    append!(u, reduce(vcat,[E3(x_i, y_f), e3, E3(x_f, y_f)]))

    U = spzeros(N, N)
    for j = 1:N
        for i = 1:N
            U[i,j] = u[i + (j - 1)*N]
        end
    end

    U_full = collect(U)

    return U_full
end

# right side
function f_5(x,y)
    return 0
end

# boundary functions
function E1_5(x,y)
    return 0
end

function E2_5(x,y)
    return 200 .* y
end

function E3_5(x,y)
    return 200 .* x
end

function E4_5(x,y)
    return 0
end

# problem specifics
E_5 = [E1_5, E2_5, E3_5, E4_5]
N_5 = 1000
Ix_5 = [0, 0.5] # range of x
Iy_5 = [0, 0.5] # range of y
I_5 = [Ix_5, Iy_5]

# solution
U = poisson_square(f_5, N_5, I_5, E_5)

# visualisation
x_5 = range(Ix_5[1], stop = Ix_5[2], length = N_5)
y_5 = range(Iy_5[1], stop = Iy_5[2], length = N_5)

figure5 = heatmap(x_5, y_5, U,
    xlabel = L"x",
    ylabel= L"y",
    title = "Exercise 5"
)

savefig(figure5, "C:\\Users\\miles\\Desktop\\numerical differential equations\\homework\\hw7\\figure5.pdf")

# error analysis
U_real = spzeros(N_5, N_5)
for i = 1:N_5
    for j = 1:N_5
        U_real[i,j] = 400*x_5[i]*y_5[j]
    end
end

error_5 = norm(U_real - U)
println(error_5)


# Exercise 6

# right side
function f_6(x,y)
    q = 6.2802
    K = 4.3543
    return - q/K
end

# boundary functions
function E1_6(x,y)
    return x * (6 - x)
end

function E2_6(x,y)
    return 0
end

function E3_6(x,y)
    return 0
end

function E4_6(x,y)
    return  y * (5 - y)
end

# problem specifics
E_6 = [E1_6, E2_6, E3_6, E4_6]
N_6_1 = 16
N_6_2 = 1000
Ix_6 = [0, 6] # range of x
Iy_6 = [0, 5] # range of y
I_6 = [Ix_6, Iy_6]

# solution
U_6_1 = poisson_square(f_6, N_6_1, I_6, E_6)
U_6_2 = poisson_square(f_6, N_6_2, I_6, E_6)

# visualisation
x_6_1 = range(Ix_6[1], stop = Ix_6[2], length = N_6_1)
y_6_1 = range(Iy_6[1], stop = Iy_6[2], length = N_6_1)

x_6_2 = range(Ix_6[1], stop = Ix_6[2], length = N_6_2)
y_6_2 = range(Iy_6[1], stop = Iy_6[2], length = N_6_2)

figure6a = heatmap(x_6_1, y_6_1, U_6_1,
    xlabel = L"x",
    ylabel= L"y",
    title = "Exercise 6, N = 16"
)

figure6b = heatmap(x_6_2, y_6_2, U_6_2,
    xlabel = L"x",
    ylabel= L"y",
    title = "Exercise 6, N = 1000"
)

savefig(figure6a, "C:\\Users\\miles\\Desktop\\numerical differential equations\\homework\\hw7\\figure6a.pdf")

savefig(figure6b, "C:\\Users\\miles\\Desktop\\numerical differential equations\\homework\\hw7\\figure6b.pdf")


# Exercise 7
# part a

function shooting_function(x, y, var)
    N = size(y, 1)
    T = var[:, 1:N]
    g = var[:, N + 1]
    dy = T*y + g

    return dy
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

function system_shooting(func, a, b, b_c, n, var)
    # function parameters
    N = size(var, 1)
    var1 = var
    var2 = hcat(var[:,1:N], zeros(N,1))

    # boundary conditions
    A = b_c[:, 1:N]
    B = b_c[:, (N+1):(2*N)]
    c = b_c[:, 2*N + 1]

    # homogenous solution
    yh_0 = Matrix{Float64}(I, N, N)
    yh = Matrix{Float64}(undef, 0, n+1)
    yh_b = zeros(N,N)

    for i = 1:N
        yh_i = RK4class(func, a, b, yh_0[:,i], n, var2)

        yh = vcat(yh, yh_i[2])
        yh_b[:,i] = yh_i[2][:,n+1]
    end
    
    # particular solution, y(a) = 0
    yp_0 = zeros(N, 1)
    yp = RK4class(func, a, b, yp_0, n, var1)
    yp_b = yp[2][:,n+1]

    # solve for s
    s = - (A .+ B*yh_b) \ (B*yp_b .- c)

    s_f = Matrix{Float64}(undef, 0, N) # make s usable
    for j = 1:N
        temp = s[j] .* Matrix{Float64}(I, N, N)
        s_f = vcat(s_f, temp)
    end

    y = s_f' * yh .+ yp[2]
    x = yp[1]
 
    return [x, y]
end

# part c

# function parameters
T_7 = [[1 3 0.4]; [0 2 0]; [5 3 1]]
g_7 = [0; -2; -0.5]

# boundary conditions
A_7 = Matrix{Float64}([[0 0 1]; [1 1 0]; [0 0 0]])
B_7 = Matrix{Float64}([[0 0 1]; [0 0 0]; [1 0 0]])
c_7 = Vector{Float64}([1; -1; 2])

# interval
a = 0.0
b = 5.0

# time steps
n = 10000

# solution
sol = system_shooting(shooting_function, a, b, hcat(A_7, B_7, c_7), n, hcat(T_7, g_7))

# plot
figure7 = plot(sol[1], sol[2][1,:],
    title = "Shooting method",
    xlabel = L"x",
    ylabel = L"y", 
    label = L"y_1(x)",
    linewidth = 2
)
plot!(figure7, sol[1], sol[2][2,:],
    label = L"y_2(x)",
    linewidth = 2
)
plot!(figure7, sol[1], sol[2][3,:],
    label = L"y_3(x)",
    linewidth = 2
)

savefig(figure7, "C:\\Users\\miles\\Desktop\\numerical differential equations\\homework\\hw7\\figure7.pdf")