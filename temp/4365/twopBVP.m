function y = twopBVP(fvec, alpha, beta, L, N)
    % fvec: values of f
    % y(0) = alpha, y(L) = beta
    % N: number of interior points

    % discrete Laplace operator
    vec1 = ones(N, 1); 
    A = spdiags([vec1 -2*vec1 vec1], -1:1, N, N);

    % construct right side
    h = L/(N + 1);

    % boundary conditions
    bc = sparse(N, 1);
    bc(1) = alpha;
    bc(N) = beta;

    b = h^2 .* fvec - bc;

    % solve the system
    y = A \ b;

    % add the boundary conditions
    y = [alpha; y; beta];
    
end