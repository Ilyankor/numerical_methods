function [x,u] = ADVECTION_DIFFUSION(epsilon,beta,n,method)
    h = 1/n; % step size
    Pe = (abs(beta)*h) / (2*epsilon); % Peclet number
    N = n-1; % interior points
    
    % select method
    switch method
        case 1 % central difference
            eps_h = epsilon;
        case 2 % upwind
            eps_h = epsilon*(1 + Pe);
        case 3 % Scharfetter-Gummel
            eps_h = epsilon*(Pe + (2*Pe) / (exp(2*Pe) - 1));
    end

    % centered difference approximations
    A_1 = -2*diag(ones(1,N)) + diag(ones(1,N-1),1) + diag(ones(1,N-1),-1);
    A_2 = diag(ones(1,N-1),1) + (-1)*diag(ones(1,N-1),-1);
    
    % set up system
    A = (-eps_h)*A_1 + ((beta*h)/2)*A_2;
    b = zeros(N,1);
    b(end) = eps_h - (beta*h)/2;
    
    % solve
    u = A\b; 
    u = [0; u; 1]; % add boundary points in
    x = linspace(0,1,n+1)';
end