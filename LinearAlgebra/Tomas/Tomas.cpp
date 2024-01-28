//
// Created by Lee on 2023/12/13.
//
#include "Tomas.h"

template<typename T>
vector<T> &resize(vector<T> &v, size_t &size) {
    // here we can't use 'const'
    auto shape = v.size();
    if (shape < size) {
        for (int i = shape; i < size; i++) {
            v.push_back(0);
        }
    } else {
        for (int i = shape; i > size; i--) {
            v.pop_back();
        }
    }
    return v;
};

// add two vectors
template<typename T, typename T1>
vector<double> operator+(vector<T> &v1, vector<T1> &v2) {
    // here we can;t use 'const'
    // the 'resize' function need change the size of the vector
    auto size = max(v1.size(), v2.size());
    resize(v1, size);
    resize(v2, size);
    vector<double> ans;
    for (int i = 0; i < size; ++i) {
        ans.push_back(v1[i] + v2[i]);
    }
    return ans;
}

// add two matrix
template<typename T>
vector<vector<T>> operator+(vector<vector<T>> &m1, vector<vector<T>> &m2) {
    auto R1 = m1.size(), C1 = m1[0].size();
    auto R2 = m2.size(), C2 = m2[0].size();
    vector<vector<T>> ans(max(R1, R2), vector<T>(max(C1, C2), 0));
    for (int i = 0; i < R1; ++i) {
        for (int j = 0; j < C1; ++j) {
            ans[i][j] += m1[i][j];
        }
    }
    for (int i = 0; i < R2; ++i) {
        for (int j = 0; j < C2; ++j) {
            ans[i][j] += m2[i][j];
        }
    }
    return ans;
}

template<typename T, typename T1>
vector<vector<double>> operator+(vector<vector<T>> &m1, vector<vector<T1>> &m2) {
    auto R1 = m1.size(), C1 = m1[0].size();
    auto R2 = m2.size(), C2 = m2[0].size();
    vector<vector<double>> ans(max(R1, R2), vector<T>(max(C1, C2), 0));
    for (int i = 0; i < R1; ++i) {
        for (int j = 0; j < C1; ++j) {
            ans[i][j] += m1[i][j];
        }
    }
    for (int i = 0; i < R2; ++i) {
        for (int j = 0; j < C2; ++j) {
            ans[i][j] += m2[i][j];
        }
    }
    return ans;
}

// add a vector and a number -- numpy like
template<typename T>
vector<T> operator+(vector<T> &v1, const T &n) {
    vector<T> ans;
    for (auto &num: v1) {
        ans.push_back(num + n);
    }
    return ans;
}

template<typename T>
vector<T> operator+(const T &n, vector<T> &v1) {
    vector<T> ans;
    for (auto &num: v1) {
        ans.push_back(num + n);
    }
    return ans;
}

template<typename T, typename T1>
vector<T> operator+(vector<T> &v1, const T1 &n) {
    vector<T> ans;
    for (auto &num: v1) {
        ans.push_back(num + n);
    }
    return ans;
}

template<typename T, typename T1>
vector<T> operator+(const T1 &n, vector<T> &v1) {
    vector<T> ans;
    for (auto num: v1) {
        ans.push_back(num + n);
    }
    return ans;
}

// multiple a vector and a number
template<typename T>
vector<T> operator*(vector<T> &v1, const T &n) {
    vector<T> ans;
    for (auto num: v1) {
        ans.push_back(num * n);
    }
    return ans;
}

template<typename T>
vector<T> operator*(const T &n, vector<T> &v1) {
    vector<T> ans;
    for (auto num: v1) {
        ans.push_back(num * n);
    }
    return ans;
}

template<typename T, typename T1>
vector<double> operator*(vector<T> &v1, const T1 &n) {
    vector<double> ans;
    for (auto num: v1) {
        ans.push_back(num * n);
    }
    return ans;
}

template<typename T, typename T1>
vector<double> operator*(const T1 &n, vector<T> &v1) {
    vector<double> ans;
    for (auto num: v1) {
        ans.push_back(num * n);
    }
    return ans;
}

// multiply of two matrix
template<typename T, typename T1>
vector<vector<double>> operator*(vector<vector<T>> &m1, vector<vector<T1>> &m2) {
    auto R1 = m1.size(), C1 = m1[0].size();
    auto R2 = m2.size(), C2 = m2[0].size();
    if (C1 != R2) {
        cout << "mismatched shape" << endl;
        exit(1);
    }
    vector<vector<double>> ans(R1, vector<double>(C2, 0));
    for (int r = 0; r < R1; r++) {
        for (int c = 0; c < C2; c++) {
            for (int i = 0; i < C1; ++i) {
                ans[r][c] += m1[r][i] * m2[i][c];
            }
        }
    }
    return ans;
}

// multiply a matrix and a vector
// here we assume it's a column vector
template<typename T, typename T1>
vector<double> operator*(vector<vector<T>> &m, vector<T1> &v) {
    auto R = m.size(), C = m[0].size();
    auto N = v.size();
    if (C != N) {
        cout << "mismatched shape" << endl;
        exit(1);
    }
    vector<double> ans(R, 0);
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            ans[i] += m[i][j] * v[j];
        }
    }
    return ans;
}

// show the vector(matrix)
template<typename T>
ostream &operator<<(ostream &os, const vector<T> &t) {
    for (auto &n: t) os << "  " << n;
    os << endl;
    return os;
}

// show the matrix
template<typename T>
ostream &operator<<(std::ostream &os, const Tomas<T> &t) {
    t.showMatrix(os);
    return os;
};
