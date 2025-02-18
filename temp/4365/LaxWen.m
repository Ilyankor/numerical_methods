function u_new = LaxWen(u_old, T)
    N = length(u_old);
    u_new = zeros(N,1);
    
    u_new(1:N-1) = u_old(1:N-1) + T*u_old(1:N-1);
    u_new(N) = u_new(1);
end