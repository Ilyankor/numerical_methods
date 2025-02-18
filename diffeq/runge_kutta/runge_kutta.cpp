#include <array>
using namespace std;

array<double, 4> rk4(function func, array<double, 4> y, array<double, 1> var) {
    array<double, 4> U = {y[1], y[2], 3.0*y[0] - y[3] + pow(t, 2.0), 2.0*y[0] + 4.0*y[3] + pow(t, 3.0) + 1};
    return U;
}