#include "SORI.h"

#define ERROR 1e-10
#define MAXITER 100

template<typename T>
bool all_closed(vector<T> &m1, vector<T> &m2)
{
    auto N = m1.size();
    for (int i = 0; i < N; ++i) {
        if (static_cast<double>(abs(m1[i] - m2[i])) > ERROR) {
            return false;
        }
    }
    return true;
}

void SOR::solve() noexcept
{
    for (int iter = 1; iter <= MAXITER; ++iter) {
        vector<double> x_new(N_, 0);
        cout << "Iteration " << iter << ":\n" << x_ << "\n";
        //evolution
        for (size_t row = 0; row < N_; ++row) {
            double s1{0}, s2{0};
            for (int i = 0; i < row; ++i) {
                s1 += A_[row][i] * x_new[i];
            }
            for (size_t i = row + 1; i < N_; ++i) {
                s2 += A_[row][i] * x_[i];
            }
            x_new[row] = (b_[row] - s1 - s2) / A_[row][row];
        }
        // check is converged
        if (all_closed<double>(x_, x_new)) {
            break;
        }
        x_ = x_new;
    }
}

vector<double> SOR::error() noexcept
{
    vector<double> err;
    err = dot(A_, x_);
    for (size_t i = 0; i < err.size(); ++i) {
        err[i] = abs(err[i] - b_[i]);
    }
    return err;
}

template<typename T>
ostream &operator<<(ostream &os, const vector<vector<T>> &mat)
{
    auto N = mat.size();
    cout << "(\n";
    for (auto &vec: mat) {
        os << vec;
    }
    cout << ")";
    return os;
}

template<typename T>
ostream &operator<<(ostream &os, const vector<T> &vec)
{
    auto N = vec.size();
    cout << "( ";
    for (int i = 0; i < N; ++i) {
        cout << vec[i] << " ";
    }
    cout << ")\n";
    return os;
}

template<typename T>
double dot(vector<T> &v1, vector<T> &v2)
{
    if (v1.size() != v2.size()) {
        cout << "shape mismatch" << endl;
        ::exit(1);
    }
    double ans{0};
    for (size_t i = 0; i < v1.size(); ++i) {
        ans += v1[i] * v2[i];
    }
    return ans;
}

template<typename T>
vector<double> dot(vector<vector<T>> &v1, vector<T> &v2)
{
    if (v1[0].size() != v2.size()) {
        cout << "shape mismatch" << endl;
        ::exit(1);
    }
    vector<double> ans(v1.size(), 0);
    for (size_t i = 0; i < v1.size(); ++i) {
        for (size_t j = 0; j < v1[0].size(); ++j) {
            ans[i] += v1[i][j] * v2[j];
        }
    }
    return ans;
}

template<typename T>
vector<double> dot(vector<T> &v1, vector<vector<T>> &v2)
{
    if (v1.size() != v2.size()) {
        cout << "shape mismatch" << endl;
        ::exit(1);
    }
    vector<double> ans(v2[0].size(), 0);
    for (size_t c = 0; c < v2[0].size(); ++c) {
        for (size_t r = 0; r < v2.size(); ++r) {
            ans[c] += v1[r] * v2[r][c];
        }
    }
    return ans;
}