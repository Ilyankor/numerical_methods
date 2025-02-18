#include <random>
#include <vector>
#include <iostream>

// external Fortran functions
extern "C"
{
    void init_q6(double* A, double* b, double* x0, const int* m, const int* n, const int* niter);
    void init_q7(double* c, double* A, double* b, double* x0, const double* rho, const int* m, const int* n, const int* niter);
}

// query question number
int getQuestion()
{
    std::cout << "\nType a 6 or a 7 to do problem 6 or 7, or type 0 to exit: ";

    int num{ };
    std::cin >> num;

    return num;
}


// create an array with random entries from N(0,1)
std::vector<double> randomMatrix(const int m, const int n)
{
    // set up random generator and normal distribution N(0,1)
    constexpr double mean{ 0.0 };
    constexpr double stddev{ 1.0 };

    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<double> normal(mean, stddev);

    const int num_entries{ m * n };             // number of entries
    std::vector<double> entries(num_entries);   // initialize entries

    // generate values
    for (auto& entry : entries)
    {
        entry = normal(generator);
    }

    return entries;
}


// create an array with random entries from U(min, max)
std::vector<double> randomMatrix(const int m, const int n, const double min, const double max)
{
    // set up random generator and uniform distribution U[min, max]
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> uniform(min, max);

    const int num_entries{ m * n };             // number of entries
    std::vector<double> entries(num_entries);   // initialize entries

    // generate values
    for (auto& entry : entries)
    {
        entry = uniform(generator);
    }

    return entries;
}


int main()
{
    std::cout << "HOMEWORK 1\n";

    int m{ };                   // rows
    int n{ };                   // cols
    int niter{ };               // num iterations
    double rho{ };              // aug Lagrangian param

    std::vector<double> v{ };   // random matrix
    std::vector<double> x0{ };  // initial guess
    std::vector<double> b{ };   // rhs
    std::vector<double> c{ };   // random vector

    while (true)
    {
        int prob_num{ getQuestion() };

        switch (prob_num)
        {
        case 6:
            std::cout << "Problem 6\n";
            
            m = 10;
            n = 1000;
            niter = 3000;

            v = randomMatrix(m, n);
            b = randomMatrix(m, 1, 0.0, 1.0);
            x0 = randomMatrix(n, 1);
            
            // projected subgradient
            init_q6(v.data(), b.data(), x0.data(), &m, &n, &niter);

            // graph results
            system("python src/hw1.py 6");
            
            break;

        case 7:
            std::cout << "Problem 7\n";

            m = 20;
            n = 200;
            niter = 2500;
            rho = 1.0e4;

            c = randomMatrix(n, 1);
            v = randomMatrix(m, n);
            b = randomMatrix(m, 1);
            x0 = randomMatrix(n, 1);
            
            // primal dual subgradient
            init_q7(c.data(), v.data(), b.data(), x0.data(), &rho, &m, &n, &niter);

            // graph results
            system("python src/hw1.py 7");

            break;

        case 0:
            std::cout << "Exiting.\n";

            return 0;

        default:
            std::cout << "Not a valid entry.\n";

            break;
        }
    }

    return 0;
}
