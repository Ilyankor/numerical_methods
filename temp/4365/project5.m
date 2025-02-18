%% Part 1.1: Diffusion equation with Euler method

clear variables

% exact solutions
g_a = @(x) sin(pi*x) - sin(3*pi*x);
u_a = @(x,t) exp(-(pi)^2*t)*sin(pi*x) - exp(-(3*pi)^2*t)*sin(3*pi*x);

% equidistant grid 
N = 30; % interior points
x = linspace(0, 1, N+2)';
dx = 1/(N+1);

% spatial discretization
vec = ones(N, 1); 
T = spdiags([vec -2*vec vec], -1:1, N, N);
Tdx = 1/(dx^2)*T;

% temporal discretization
t_end = 0.1;
M_i = [200, 173, 170]; % different time steps

for i = 1:length(M_i)
    % equidistant temporal grid
    M = M_i(i);
    t = linspace(0, t_end, M+1);
    dt = t_end/M;

    % initialize variables
    u = zeros(N+2, M+1); % computed solution
    u_exact = zeros(N+2, M+1); % exact solution
    err = zeros(M+1, 1); % error

    % initial condition
    u(:,1)= g_a(x);
    u_exact(:,1) = u_a(x,t(1));
    err(1) = norm(u_exact(:,1) - u(:,1));

    % solve the equation
    for m = 1:M
        u(:,m+1) = eulerstep(Tdx, u(:,m), dt); % explicit Euler
        u_exact(:,m+1) = u_a(x,t(m+1));
        err(m+1) = norm(u_exact(:,m+1) - u(:,m+1));
    end

    % visualization
    [T,X] = meshgrid(t,x);

    % Euler solution plot
    figure(1 + 3*(i - 1))
    mesh(T, X, u)
    title('Computed solution')
    xlabel('t') 
    ylabel('x')
    zlabel('u(x,t)')
    
    % exact solution plot
    figure(2 + 3*(i - 1))
    mesh(T, X, u_exact)
    title('Exact solution')
    xlabel('t') 
    ylabel('x')
    zlabel('u(x,t)')

    % error plot
    figure(3 + 3*(i - 1))
    plot(t, err, LineWidth = 2)
    title('Error over time')
    xlabel('t') 
    ylabel('error')
    
    % CFL condition
    CFL = dt/dx^2;
    norm_0 = norm(u(:,M));
    norm_1 = norm(u(:,M+1));

    disp('The CFL number is:')
    disp(CFL)
    
    if (CFL < 1/2)
        disp('CFL condition is met and the solution should be stable')
    elseif (CFL > 1/2 && norm_1 < norm_0)
        disp('CFL condition is not met and the solution is stable')
    elseif (CFL>1/2 && norm_1 > norm_0)
        disp('CFL condition is not met and the solution is not stable')
    end
end

%% Part 1.2: Diffusion equation with Crank-Nicolson method

clear variables

% exact solutions
g_a = @(x) sin(pi*x) - sin(3*pi*x);
u_a = @(x,t) exp(-(pi)^2*t)*sin(pi*x) - exp(-(3*pi)^2*t)*sin(3*pi*x);

% test on another initial condition
%g_a = @(x) (1 - x) .* x.^2 .* exp(x);

% equidistant grid
N = 30;
x = linspace(0, 1, N+2)';
dx = 1/(N+1);

% spatial discretization
vec = ones(N, 1); 
T = spdiags([vec -2*vec vec], -1:1, N, N);
Tdx = 1/(dx^2)*T;

% temporal discretization
t_end = 0.1;
M_i = [200, 170, 40];

for i = 1:length(M_i)
    % equidistant temporal grid
    M = M_i(i);
    t = linspace(0, t_end, M+1);
    dt = t_end/M;

    % initialize variables
    u = zeros(N+2, M+1); % computed solution
    u_exact = zeros(N+2, M+1); % exact solution
    err = zeros(M+1, 1); % error

    % initial condition
    u(:,1)= g_a(x);
    u_exact(:,1) = u_a(x,t(1));
    err(1) = norm(u_exact(:,1) - u(:,1));

    % solve the equation
    for m = 1:M
        u(:,m+1) = TRstep(Tdx, u(:,m), dt); % Crank-Nicolson
        u_exact(:,m+1) = u_a(x,t(m+1));
        err(m+1) = norm(u_exact(:,m+1) - u(:,m+1));
    end

    % visualization
    [T,X] = meshgrid(t,x);

    % Euler solution plot
    figure(1 + 3*(i - 1) + 9)
    mesh(T, X, u)
    title('Computed solution')
    xlabel('t') 
    ylabel('x')
    zlabel('u(x,t)')
    
    % exact solution plot
    figure(2 + 3*(i - 1) + 9)
    mesh(T, X, u_exact)
    title('Exact solution')
    xlabel('t') 
    ylabel('x')
    zlabel('u(x,t)')

    % error plot
    figure(3 + 3*(i - 1) + 9)
    plot(t, err, LineWidth = 2)
    title('Error over time')
    xlabel('t') 
    ylabel('error')
end

%% Part 2: Advection equation

clear variables

% initial condition
g_a = @(x) exp(-100*(x-0.5).^2);

% discretization numbers
N_i = [90, 81];
M_i = [90, 100];

% L2 norm plot legend
leg = ["a*mu = 1", "a*mu = -1", "dx = 1/81", "dt = 1/20"];

% test for various options
for i = 1:4
    % options
    if i == 1 % a*mu = 1, a > 0
        a = 0.2;
        N = N_i(1);
        M = M_i(1);
    elseif i == 2 % a*mu = -1, a < 0
        a = -0.2;
        N = N_i(1);
        M = M_i(1);
    elseif i == 3 % a*mu = 0.9, dx = 1/243
        a = 0.2;
        N = N_i(2);
        M = M_i(1);
    else % a*mu = 0.9, dt = 1/300
        a = 0.2;
        N = N_i(1);
        M = M_i(2);
    end

    % spatial discretization
    x = linspace(0, 1, N+1);
    dx = 1/N;

    % temporal discretization
    t_end = 5;
    t = linspace(0, t_end, M+1);
    dt = t_end/M;

    % Lax-Wendroff
    amu = a*dt/dx;
    beta = amu/2*(amu+1);
    alpha = -amu^2;
    gamma = amu/2*(amu-1);

    % matrix
    vec = ones(N, 1); 
    A = spdiags([beta*vec alpha*vec gamma*vec], -1:1, N, N);
    A(1,N) = beta;
    A(N,1) = gamma;

    % initialize
    U = zeros(N+1, M+1); % initialize variable
    U(:,1) = g_a(x); % initial condition

    % create video
    v = figure(24);
    v.Units = 'pixels';
    v.Position = [0 0 1920 1080];
    str = strcat('AdvectionMovie',num2str(i),'.avi');
    vidObj = VideoWriter(str, 'Motion JPEG AVI');
    vidObj.FrameRate = M/t_end;
    open(vidObj);

    for m = 1:M
        U(:,m+1) = LaxWen(U(:,m), A); % Lax-Wendroff
        plot(x, U(:,m), LineWidth = 2)
        title('Solution through time')
        xlabel('x')
        ylabel('u(x,t_i)')

        % write video
        currFrame = getframe(gcf);
        writeVideo(vidObj,currFrame);
    end

    close(vidObj);

    % visualization
    [T,X] = meshgrid(t,x);

    % plot solution
    figure(19 + (i-1))
    mesh(T,X,U);
    str2 = strcat('amu = ', num2str(amu), ...
        ', dx = ', num2str(dx,2), ...
        ', dt = ', num2str(dt,2));
    title(str2)
    xlabel('t') 
    ylabel('x')
    zlabel('u(x,t)')

    % plot rms
    rms_sol = zeros(M+1,1);
    for j = 1:M+1
        rms_sol(j) = rms(U(:,j));
    end
    
    figure(23);
    ax = gca; 
    hold on
    plot(ax, t, rms_sol, 'DisplayName', leg(i), LineWidth = 2)
    title('Root mean squared')
    xlabel('t') 
    ylabel('rms(u(x,t_i)')
    hold off

    legend
end

%% export all figures
for j = 1:23
    h = figure(j);
    set(h,'Units','Inches');
    pos = get(h,'Position');
    set(h,'PaperPositionMode','Auto', ...
        'PaperUnits','Inches', ...
        'PaperSize',[pos(3), pos(4)])
    str = strcat('figure',num2str(get(gcf,'Number')),'.pdf');
    print('-vector','-dpdf',str);
end