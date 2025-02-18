function u_new = eulerstep(Tdx, u_old, dt)
    N = length(u_old);
    u_new = zeros(N,1);

    u_new(2:N-1) = u_old(2:N-1) + dt*Tdx*u_old(2:N-1);
end