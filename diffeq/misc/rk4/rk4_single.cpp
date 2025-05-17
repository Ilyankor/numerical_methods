#include <functional>
#include <fstream>
#include <iostream>
#include <cmath>


double func(double t, double y) {
    return y - t - pow(y, 0.85);
}


double rk4(std::function<double (double, double)> f, double t_0, double y_0, double h, int n) {
    // file writer
    std::ofstream outf{ "results.csv" };

    // initialize f(t0, y0)
    double t = t_0;
    double y = y_0;

    // write initial values
    outf << t << "," << y << "\n";

    // Runge-Kutta 4th order
    for (int i{ 0 }; i < n; ++i) {

        double k_1 = f(t, y);
        double k_2 = f(t + 0.5*h, y + 0.5*h*k_1);
        double k_3 = f(t + 0.5*h, y + 0.5*h*k_2);
        double k_4 = f(t + h, y + h*k_3);

        y = y + (1.0/6.0)*h*(k_1 + 2.0*k_2 + 2.0*k_3 + k_4);
        t = t + h;

        // write to file
        outf << t << "," << y << "\n";
    }

    return y;
}

// int main() {
//     rk4(func, 1.0, 3.5, 0.001, 2000);
//     return 0;
// }