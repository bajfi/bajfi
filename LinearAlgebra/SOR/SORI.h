#ifndef SOR_SORI_H
#define SOR_SORI_H

#include "SOR.h"

SOR::SOR(size_t N) :
        N_(N),
        A_(N_, vector<double>(N_, 0)),
        b_(N_, 0),
        x_(N_, 0)
{}

SOR::SOR(vector<vector<double>> &A, vector<double> &b) :
        A_(A),
        b_(b),
        N_(A.size()),
        x_(N_, 0)
{}

SOR::SOR(vector<vector<double>> &A, vector<double> &b, vector<double> &x) :
        A_(A),
        b_(b),
        N_(A.size()),
        x_(x)
{}

ostream &operator<<(ostream &os, const SOR &sor)
{
    auto N = sor.size();
    auto A = sor.mat();
    auto b = sor.b();
    cout << "{\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N - 1; ++j) {
            cout << A[i][j] << "*x" << i + 1
                 << " + ";
        }
        cout << A[i].back() << "*x" << N
             << " = " << b[i] << "\n";
    }
    cout << "}\n";
    return os;
};

#endif //SOR_SORI_H
