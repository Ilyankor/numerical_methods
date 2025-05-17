#include <iostream>
#include <array>
#include <cmath>
using namespace std;

array<double, 4> test_func(double t, array<double, 4> y, array<double, 1> var) {
    array<double, 4> U = {y[1], y[2], 3.0*y[0] - y[3] + pow(t, 2.0), 2.0*y[0] + 4.0*y[3] + pow(t, 3.0) + 1};
    return U;
}

int main() {
    double t = 2.0;
    array<double, 4> y = {0.2, -1.5, 0.0, 4};
    array<double, 1> var = {0.0};

    array<double, 4> u = test_func(t, y, var);

    for (int i = 0; i < 4; i++) { 
        cout << u[i] << "\n";
    }
}