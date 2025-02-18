module hw1
    use, intrinsic :: iso_c_binding, only: dp => c_double, c_int
    implicit none

contains

    subroutine init_q6(A, b, x0, m, n, niter) bind(C, name="init_q6")
        ! Executes question 6.

        integer(c_int), intent(in)                  :: m, n, niter  ! rows, cols, num iterations
        real(dp), dimension(m, n), intent(inout)    :: A            ! mxn matrix A
        real(dp), dimension(m), intent(in)          :: b            ! rhs
        real(dp), dimension(n), intent(inout)       :: x0           ! initial guess
        real(dp), dimension(n, niter+1)             :: x            ! store results
        
        ! apply projected subgradient
        call proj_subgradient(A, b, x0, m, n, niter, x)

        ! write information
        call write_info(A, m, n, "./out/6/A.dat")
        call write_info(b, m, 1, "./out/6/b.dat")
        call write_info(x0, n, 1, "./out/6/x0.dat")
        call write_info(x, n, niter+1, "./out/6/x.dat")

    end subroutine init_q6


    subroutine init_q7(c, A, b, x0, m, n, rho, niter) bind(C, name="init_q7")
        ! Executes question 7.

        integer(c_int), intent(in)              :: m, n, niter  ! rows, cols, num iterations
        real(dp), dimension(n), intent(in)      :: c            ! objective vector
        real(dp), dimension(m, n), intent(in)   :: A            ! mxn matrix A
        real(dp), dimension(m), intent(in)      :: b            ! rhs
        real(dp), dimension(n), intent(in)      :: x0           ! initial guess
        real(dp), intent(in)                    :: rho          ! aug Lagrangian param

        real(dp), dimension(n, niter+1)         :: x            ! store x results
        real(dp), dimension(m, niter+1)         :: lam          ! store Lagrangian results

        ! apply primal dual subgradient
        call primal_dual(c, A, b, x0, m, n, rho, niter, x, lam)

        ! write information
        call write_info(c, n, 1, "./out/7/c.dat")
        call write_info(A, m, n, "./out/7/A.dat")
        call write_info(b, m, 1, "./out/7/b.dat")
        call write_info(x0, n, 1, "./out/7/x0.dat")
        call write_info(x, n, niter+1, "./out/7/x.dat")
        call write_info(lam, m, niter+1, "./out/7/lam.dat")

    end subroutine init_q7


    subroutine write_info(A, m, n, name)
        ! Writes an mxn array to a binary file.

        integer(c_int), intent(in)              :: m, n     ! dimensions
        real(dp), intent(in), dimension(m, n)   :: A        ! matrix
        integer                                 :: io       ! file unit integer
        character(len=*), intent(in)            :: name     ! file name
        
        ! write array to file
        open(newunit=io, file=trim(name), form="unformatted", access="stream", status="replace", action="write")
        write(io) A
        close(io)

    end subroutine write_info

    
    subroutine proj_subgradient(A, b, x0, m, n, niter, x)
        ! Applies the projected subgradient descent method to the problem
        ! minimize |x|_1 (l1 norm)
        ! subject to Ax = b
        ! where A is mxn with m < n and rank(A) = m

        integer(c_int), intent(in)              :: m, n, niter  ! rows, cols, num iterations
        real(dp), dimension(m, n), intent(in)   :: A            ! mxn matrix A
        real(dp), dimension(m), intent(in)      :: b            ! rhs
        real(dp), dimension(n), intent(inout)   :: x0           ! initial guess

        real(dp), dimension(n)                  :: g            ! subgradient
        real(dp), dimension(n, niter+1), intent(out) :: x       ! store results

        real(dp), dimension(n, m)               :: AT           ! A^T
        real(dp), dimension(m, m)               :: AAT          ! A*A^T
        real(dp), dimension(n)                  :: xk           ! xk
        real(dp), dimension(m)                  :: y            ! y = (A*A^T)^-1Ag
    
        integer                                 :: i            ! iterator
        integer, dimension(m)                   :: ipiv         ! for dgesv
        integer                                 :: info         ! for dgesv

        ! store computations
        AT = transpose(A)
        AAT = matmul(A, AT)

        ! project initial guess
        y = matmul(A, x0) - b                           ! y = Ax0 - b
        call dgesv(m, 1, AAT, m, ipiv, y, m, info)      ! y = (A*A^T)^-1*(Ax0 - b)
        x0 = x0 - matmul(AT, y)                         ! x0 = x0 - A^Ty
        
        xk = x0                                         ! set xk to projected x0
        x(:, 1) = x0                                    ! store result
        
        ! projected subgradient method with step size 1/i
        do i = 1, niter
            g = sign(1.0_dp, xk)                                ! subgradient g in df(xk)

            y = matmul(A, g)                                    ! y = Ag
            call dgetrs('N', m, 1, AAT, m, ipiv, y, m, info)    ! y = (A*A^T)^-1*(Ag)
            
            xk = xk - (1.0_dp / real(i, dp)) * (g - matmul(AT, y))  ! xk = xk - t(g - A^Ty)
            x(:, i+1) = xk                                      ! store result

        end do
    
    end subroutine proj_subgradient


    subroutine primal_dual(c, A, b, x0, m, n, rho, niter, x, lam)
        ! Applies the primal dual subgradient descent method to the problem
        ! minimize c*x
        ! subject to Ax <= b

        integer(c_int), intent(in)              :: m, n, niter  ! rows, cols, num iterations
        real(dp), dimension(n), intent(in)      :: c            ! objective vector
        real(dp), dimension(m, n), intent(in)   :: A            ! mxn matrix A
        real(dp), dimension(m), intent(in)      :: b            ! rhs
        real(dp), dimension(n), intent(in)      :: x0           ! initial guess
        real(dp), intent(in)                    :: rho          ! aug Lagrangian param

        real(dp), dimension(n, niter+1), intent(out) :: x       ! store x results
        real(dp), dimension(m, niter+1), intent(out) :: lam     ! store Lagrangian results

        real(dp), dimension(n, m)               :: AT           ! A^T
        real(dp), dimension(n, m)               :: AT_copy      ! copy of A^T
        real(dp), dimension(n)                  :: xk           ! xk
        real(dp), dimension(m)                  :: lamk         ! lamdak
        real(dp), dimension(m)                  :: y            ! y = Ax - b
        real(dp), dimension(n)                  :: T1           ! KKT operator
        real(dp)                                :: T            ! norm of KKT operator
        real(dp)                                :: alph         ! step size
    
        integer                                 :: i, j         ! iterators

        ! store computations
        AT = transpose(A)
        AT_copy = AT

        ! initial guesses
        x(:, 1) = x0
        xk = x0
        lam(:, 1) = 0
        lamk = 0

        ! primal dual subgradient method
        do i = 1, niter
            y = matmul(A, xk) - b                   ! y = Ax - b
            do j = 1, m
                if (y(j) .le. 0) then
                    y(j) = 0                        ! y = (Ax - b)+
                    AT(:, j) = 0                    ! modified AT
                end if
            end do

            T1 = c + matmul(AT, lamk + rho * y)     ! first elem of KKT
            T = norm2([ norm2(T1), norm2(y) ])      ! norm of KKT
            alph = 1.0_dp / (real(i, dp) * T)       ! step size

            xk = xk - alph * T1                     ! xk+1
            lamk = lamk + alph * y                  ! lamk+1

            x(:, i+1) = xk                          ! store xk
            lam(:, i+1) = lamk                      ! store lamk

            AT = AT_copy                            ! reset AT
        end do

    end subroutine primal_dual

end module hw1
