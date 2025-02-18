function [y, y_p, d] = schrodinger(fvec, N)
    % fvec: function values
    % N: number of interior points

    % discrete Laplace operator
    vec1 = ones(N, 1); 
    T = spdiags([vec1 -2*vec1 vec1], -1:1, N, N);
    
    % potential function
    h = 1/(N + 1);
    P = h^2 .* speye(N,N) * fvec;
    
    A = T - P;

    % solve
    n = 10; % number of eigenvalues
    [V, D] = eigs(A, n, 'smallestabs');

    % output
    d = ones(n, 1);
    y = zeros(N + 2, n);
    y_p = zeros(N + 2, n);

    for i = 1:n
        % energy levels
        d(i) = - D(i, i) / h^2;

        % normalise the wave function
        y(2:N+1, i) = V(:, i)/max(abs(V(:,i)));

        % wave probability function normalised to 100
        y_p(:, i) = 50 * abs(y(:, i)).^2 + d(i);

        % wave probability function normalised to 100
        y(:, i) = 50 * y(:, i) + d(i);
    end

    % plot the wave function
    x = linspace(0, 1, N + 2);

    figure;
    plot(x,y, ...
        "LineWidth", 2)
    title('$50\hat{\psi}_k + E_k$', "Interpreter", "latex")
    xlabel('$x$', "Interpreter", "latex")
    ylabel('$\hat{\psi}_k (x)$', "Interpreter", "latex")
    legend('$\hat{\psi}_1$', '$\hat{\psi}_2$', '$\hat{\psi}_3$', ...
        '$\hat{\psi}_4$', '$\hat{\psi}_5$','$\hat{\psi}_6$', ...
        '$\hat{\psi}_7$', '$\hat{\psi}_8$', '$\hat{\psi}_9$',...
        '$\hat{\psi}_{10}$', ...
        "Interpreter", "latex")

    % plot the probability function
    figure;
    plot(x,y_p, ...
        "LineWidth", 2)
    title('$50\left|\hat{\psi}_k\right|^2 + E_k$', "Interpreter", "latex")
    xlabel('$x$', "Interpreter", "latex")
    ylabel('$\left|\hat{\psi}_k\right|^2$', "Interpreter", "latex")
    legend('$\hat{\psi}_1$', '$\hat{\psi}_2$', '$\hat{\psi}_3$', ...
        '$\hat{\psi}_4$', '$\hat{\psi}_5$','$\hat{\psi}_6$', ...
        '$\hat{\psi}_7$', '$\hat{\psi}_8$', '$\hat{\psi}_9$',...
        '$\hat{\psi}_{10}$', ...
        "Interpreter", "latex")

end
