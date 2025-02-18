module runge_kutta

contains

subroutine rk4(func, h, n, t0, y0, var, y)
    implicit none

    abstract interface
        function f(tx, x, vars) result(U)
            real(kind = 8), intent(in) :: tx
            real(kind = 8), dimension(:), intent(in) :: x
            real(kind = 8), dimension(:), intent(in) :: vars
            real(kind = 8), dimension(size(x)) :: U
        end function
    end interface

    procedure(f) :: func

    real(kind = 8), intent(in) :: h, t0
    integer(kind = 4), intent(in) :: n
    real(kind = 8), dimension(:), intent(in) :: y0
    real(kind = 8), dimension(:), intent(in) :: var

    integer(kind = 4) :: i
    real(kind = 8) :: t
    real(kind = 8), dimension(size(y0)) :: s1, s2, s3, s4, y
    
    t = t0
    y = y0

    do i = 1, n
        s1 = h * func(t, y, var)
        s2 = h * func((t + 0.5*h), (y + 0.5 * s1), var)
        s3 = h * func((t + 0.5*h), (y + 0.5 * s2), var)
        s4 = h * func(t + h, (y + s3), var)

        y = y + (s1 + 2*s2 + 2*s3 + s4) / 6.0
        t = t + h
    end do
end subroutine

end module