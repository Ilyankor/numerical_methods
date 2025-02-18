function u_new = TRstep(Tdx, u_old, dt)
    N = length(u_old);
    u_new = zeros(N,1);

    A = eye(N-2) - (dt/2)*Tdx;
    b = u_old(2:N-1) + (dt/2)*Tdx*u_old(2:N-1);
    
    u_new(2:N-1) = A\b;
end