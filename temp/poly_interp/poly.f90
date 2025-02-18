program polynomial_interpolation
    implicit none

    ! input file structure
    ! first line: degree of the polynomial, number of inputs
    ! second line: input values, comma separated
    ! third line: output values, comma separated

    ! variable declarations
    integer, parameter :: dp = selected_real_kind(15)
    integer :: i

    logical :: input_exists
    integer :: io, deg, num, dim
    real(dp), allocatable :: x(:), y(:), temp(:,:), A(:,:), b(:), err(:)

    integer :: info
    integer, allocatable :: ipiv(:)


    ! read the input file
    inquire(file="input.txt", exist=input_exists)
    if (input_exists) then
        open(newunit=io, file="input.txt", status="old", action="read")
            read(io, *) deg, num
            allocate(x(num), y(num))
            read(io, *) (x(i), i = 1, num)
            read(io, *) (y(i), i = 1, num)
        close(io)
    else
        stop "Input file does not exist."
    end if

    ! form the equation Ax = b
    dim = deg + 1
    allocate(temp(num,dim), A(dim,dim), b(dim))
    do i = 1, dim
        temp(:,i) = x**(i-1)
    end do

    A = matmul(transpose(temp), temp)
    b = matmul(transpose(temp), y)

    ! solve Ax = b using LAPACK
    allocate(ipiv(dim))
    call dgesv(dim, 1, A, dim, ipiv, b, dim, info)
    print *, "The coefficients are ", b

    ! compute error
    allocate(err(num))
    err = abs(matmul(temp, b) - y)
    ! print *, "The errors are ", err
    print *, "The norm of the error is ", norm2(err)

end program polynomial_interpolation