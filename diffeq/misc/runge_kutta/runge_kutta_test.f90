program runge_kutta_test
    use :: runge_kutta
    implicit none

    real(kind = 8) :: h, t0
    integer(kind = 4) :: n
    real(kind = 8), dimension(4) :: y0
    real(kind = 8), dimension(1) :: var
    real(kind = 8), dimension(4) :: y

    real(kind = 8) :: rate
    integer(kind = 4) :: beginning, end, i

    h = 0.001
    t0 = 0.0
    n = 1000
    y0 = (/ 0.2, 3.1, -1.2, 0.0 /)
    var = (/ 0.0 /)

    ! benchmarking
    call system_clock(beginning, rate)
    do i = 1, 1000
        call rk4(test_func, h, n, t0, y0, var, y)
    end do
    call system_clock(end)
    print *, "elapsed time: ", (real(end - beginning) / real(rate)) / 1000 * 10**6
    ! call rk4(test_func, h, n, t0, y0, var, y)
    ! print *, y

contains

    function test_func(t, x, vars) result(U)
        implicit none

        real(kind = 8), intent (in) :: t
        real(kind = 8), dimension(:), intent (in) :: x
        real(kind = 8), dimension(:), intent (in) :: vars
        real(kind = 8), dimension(size(x)) :: U

        U = (/ x(2), x(3), 3*x(1) - x(4) + t**2, 2*x(1) + 4*x(4) + t**3 + 1 + vars(1)/)
    end function

end program

! gfortran -o output runge_kutta.f90 runge_kutta_test.f90