%% Task 1.1: two point boundary value problem solver

L = 1;
N = 30000;

a = 0; % y(0) = 0
b = 0; % y(1) = 0

x = linspace(0, L, N+2);
func = sin(x); % y'' = sin(x)
f = func(2:N + 1)'; % fvec

sol = twopBVP(f, a, b, L, N);

% plot the solution
plot(x, sol, ...
    "LineWidth",2);
title('Solution of test function')
xlabel('$x$', "Interpreter", "latex")
ylabel('$y$', "Interpreter", "latex")

exportgraphics(gcf,'figure1-1a.pdf','ContentType','vector')

% compute error for various N
error = zeros(15, 2);
for i = 1:15
    n = 2^i;
    h_i = 1/(n + 1);
    error(i, 1) = h_i;

    x_i = linspace(0, L, n+2);
    func_i = sin(x_i);
    f_i = func_i(2:n + 1)';

    sol_i = twopBVP(f_i, a, b, L, n);
    exact_i = (sin(1).*x_i - sin(x_i))';

    err_i = sol_i - exact_i;
    error(i, 2) = max(abs(err_i));
end

% error visualisation
loglog(error(:,1), error(:,2), ...
    "LineWidth", 2);
title('Error vs. h')
xlabel('$h$', "Interpreter", "latex")
ylabel('error$(h)$', "Interpreter", "latex")

exportgraphics(gcf,'figure1-1b.pdf','ContentType','vector')

clearvars
clf('reset')


%% Task 1.2: the beam equation

% common parameters
a = 0;
b = 0;
L = 10;
N = 999;

x = linspace(0, L, N+2);

% first equation
% M'' = q(x)
func_1 = -50000.*ones(1, N+2);
f_1 = func_1(2:N + 1)';

% solution M(x)
sol_1 = twopBVP(f_1, a, b, L, N);

% second equation
% u'' = M(x)/(EI)
Y = 1.9E11; % Young's modulus
I_func = 1E-3 .* (3 - 2 * (cos((pi .* x)/L)).^12); % moment of inertia
func_2 = sol_1' ./ (Y .* I_func);
f_2 = func_2(2:N + 1)';

% solution
sol_2 = twopBVP(f_2, a, b, L, N);

% visualisation
plot(x, sol_2, ...
    "LineWidth",2)
title('Deflection')
xlabel('$x$', "Interpreter", "latex")
ylabel('$y$', "Interpreter", "latex")

exportgraphics(gcf,'figure1-2.pdf','ContentType','vector')

% deflection at midpoint
format long
str = "The deflection at the midpoint in mm is: ";
disp(str)
disp(sol_2(501)*1000)

clearvars
clf('reset')


%% Task 2.1: Sturm-Liouville

n = 3; % number of eigenvalues
lambda_exact = (-(pi.*(1:n)).^2)';

% compute approximate eigenvalues
error = zeros(3, 15);
N = zeros(15, 1);
for i = 1:15
    N(i) = 2^i + 1;
    h = 1/(N(i) + 1);
    
    vec1 = ones(N(i), 1); 
    T = 1/h^2 * spdiags([vec1 -2*vec1 vec1], -1:1, N(i), N(i));

    d = eigs(T, 3, 'smallestabs');

    error(:,i) = d - lambda_exact;
end

% visualisation
loglog(N, error, ...
    "LineWidth", 2)
title('Error of eigenvalues')
xlabel('$N$', "Interpreter", "latex")
ylabel('$\lambda_i$', "Interpreter", "latex")
legend('$\lambda_1$', '$\lambda_2$', '$\lambda_3$', ...
    "Interpreter", "latex")

exportgraphics(gcf,'figure2-1a.pdf','ContentType','vector')

% N = 499, eigenvalues and eigenfunctions
N_2 = 499;
h_2 = 1/(N_2 + 1);

vec_2 = ones(N_2, 1); 
T_2 = 1/h_2^2 * spdiags([vec_2 -2*vec_2 vec_2], -1:1, N_2, N_2);

% solution
[V, D] = eigs(T_2, 3, 'smallestabs');

% eigenvalues
d_2 = ones(3, 1);
for j = 1:3
    d_2(j) = D(j,j);
end

format long
str = "The first three eigenvalues are:";
disp(str)
disp(d_2)

% add in boundary conditions
V_2 = [zeros(1,3); V; zeros(1,3)];

% plot the eigenfunctions
x = linspace(0, 1, N_2 + 2);

plot(x, V_2, ...
    "LineWidth", 2)
title('Eigenfunctions')
xlabel('$x$', "Interpreter", "latex")
ylabel('$u_i$', "Interpreter", "latex")
legend('$u_1$', '$u_2$', '$u_3$', ...
    "Interpreter", "latex")

exportgraphics(gcf,'figure2-1b.pdf','ContentType','vector')

clearvars
clf('reset')


%% Task 2.2: the Schrodinger equation

% potential: V(x) = 0
N_0 = 499;
fvec_0 = zeros(N_0, 1);

[y_0, y_p_0, E_0] = schrodinger(fvec_0, N_0);

% potential: V(x) = 700(0.5 − |x − 0.5|)
N_1 = 5000;

x_1 = linspace(0, 1, N_1 + 2);
fvec_1 = 700*(0.5 - abs(x_1 - 0.5));
f_1 = fvec_1(2:N_1 + 1)';

[y_1, y_p_1, E_1] = schrodinger(f_1, N_1);

% potential: V(x) = 800sin^2(pi*x)
N_2 = 5000;

x_2 = linspace(0, 1, N_2 + 2);
fvec_2 = 800*(sin(pi .* x_2)).^2;
f_2 = fvec_2(2:N_2 + 1)';

[y_2, y_p_2, E_2] = schrodinger(f_2, N_2);

% potential: V(x) = 5 x^4 - 10 x^3 + 5.9375 x^2 - 0.9375 x
N_3 = 5000;

x_3 = linspace(0, 1, N_3 + 2);
fvec_3 = 5*x_3 .^4 - 10*x_3 .^3 + 5.9375*x_3 .^2 - 0.9375*x_3;
f_3 = fvec_3(2:N_3 + 1)';

[y_3, y_p_3, E_3] = schrodinger(f_3, N_3);

clearvars
