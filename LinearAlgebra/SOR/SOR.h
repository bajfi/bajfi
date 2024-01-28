#ifndef SOR_SOR_H
#define SOR_SOR_H

#include <cstdlib>
#include <vector>
#include <iostream>

using namespace std;

class SOR
{
    size_t N_;      // size of A matrix
    vector<vector<double>> A_;  // coefficient matrix
    vector<double> b_;      // b vector in rhs
    vector<double> x_;      // vector for answer

public:

    explicit SOR(size_t N);

    SOR(const SOR &sor) = default;

    SOR(vector<vector<double>> &A, vector<double> &b);

    SOR(vector<vector<double>> &A, vector<double> &b, vector<double> &x);

//    member functions
    [[nodiscard]] size_t size() const noexcept
    { return N_; }

    [[nodiscard]] vector<vector<double>> mat() const noexcept
    { return A_; }

    [[nodiscard]] vector<double> b() const noexcept
    { return b_; }

    [[nodiscard]] vector<double> x() const noexcept
    { return x_; }

    void solve() noexcept;

    vector<double> error() noexcept;
};

template<typename T>
ostream &operator<<(ostream &os, const vector<vector<T>> &mat);

template<typename T>
ostream &operator<<(ostream &os, const vector<T> &vec);

ostream &operator<<(ostream &os, const SOR &sor);

// dot production
template<typename T>
double dot(vector<T> &v1, vector<T> &v2);

template<typename T>
vector<double> dot(vector<vector<T>> &v1, vector<T> &v2);

template<typename T>
vector<double> dot(vector<T> &v1, vector<vector<T>> &v2);

#endif //SOR_SOR_H
