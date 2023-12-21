//
// Created by Lee on 2023/12/13.
//

#ifndef CPP_PROGRAME_TOMAS_H
#define CPP_PROGRAME_TOMAS_H
#pragma once

// Tomas method also means the chaseAfter method
// which used to solve 3 diagonal matrix
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

using namespace std;

// -------------------  pre-declaration  --------------------------------
template<typename T>
vector<T> &resize(vector<T> &v, size_t &size);

template<typename T, typename T1>
vector<T> operator+(vector<T> &v1, const T1 &n);

template<typename T, typename T1>
vector<T> operator+(const T1 &n, vector<T> &v1);

template<class T>
class Tomas {
    // ----------------------------  pribate data  -------------------------------
private:
    vector<double> ldiag_;  // lower diagonal
    vector<double> mdiag_;  // mid diagonal
    vector<double> udiag_;  // upper diagonal
    vector<double> right_udiag_;
    vector<double> right_mdiag_;
    vector<double> left_ldiag_;
    vector<double> left_mdiag_;
    vector<double> y_;      // solution of the linear function
    vector<double> yl_;      // left matrix solution of the linear function
    vector<double> yr_;      // right matrix solution of the linear function

    // solve equation
    // pos: 0 for downside triangle, 1 for upside triangle
    void solveEquation(vector<T> &mdiag, vector<T> &offset, bool pos) {
        // upper triangle
        // yr[n] = y[n]/mdiag[n]
        // yr[i] = (y[i] - udiag[i]*yr[i+1]) / mdiag[i]
        if (pos) {
            yr_.back() = yl_.back() / mdiag.back();
            for (int i = yl_.size() - 2; i >= 0; i--) {
                yr_[i] = (yl_[i] - offset[i] * yr_[i + 1]) / mdiag[i];
            }
        }
            // downside triangle
            // yl[0] = y[0]
            // yl[i] = y[i] - ldiag[i-1]*y[i-1]
        else {
            yl_[0] = y_[0];
            for (int i = 1; i < y_.size(); ++i) {
                yl_[i] = y_[i] - offset[i - 1] * yl_[i - 1];
            }
        }
    }

    // the matrix must satisfy the conditions
    bool prerequisite() {
        // |a| and |c| != 0 (not compulsory here)
        // |b| >= |a| + |c|
        for (int i = 0; i < y_.size() - 1; ++i) {
            if (abs(mdiag_[i]) < abs(ldiag_[i]) + abs(udiag_[i])) {
                cout << "The mid diagonal must greater than others" << endl;
                return false;
            }
        }
        return true;
    }


public:
    // ----------------------------- constructor  --------------------------------
    Tomas(T *ldiag, T *mdiag, T *udiag, T *y, int size) {
        for (int i = 0; i < size; i++) {
            ldiag_.push_back(ldiag[i]);
            mdiag_.push_back(mdiag[i]);
            udiag_.push_back(udiag[i]);
            y_.push_back(y[i]);
        }
    };

    // copy constructor
    Tomas(const Tomas &t) :
            ldiag_(t.ldiag_.begin(), t.ldiag_.end()),
            mdiag_(t.mdiag_.begin(), t.mdiag_.end()),
            udiag_(t.udiag_.begin(), t.udiag_.end()),
            y_(t.y_.begin(), t.y_.end()),
            yl_(t.yl_.begin(), t.yl_.end()),
            yr_(t.yr_.begin(), t.yr_.end()),
            left_ldiag_(t.left_ldiag_.begin(), t.left_ldiag_.end()),
            left_mdiag_(t.left_mdiag_.begin(), t.left_mdiag_.end()),
            right_mdiag_(t.right_mdiag_.begin(), t.right_mdiag_.end()),
            right_udiag_(t.right_udiag_.begin(), t.right_udiag_.end()) {};

    // construct with 3 diagonals
    Tomas(vector<T> &ldiag,
          vector<T> &mdiag,
          vector<T> &udiag,
          vector<T> &y) : y_(y) {
        // the size is depend on the size of y
        auto size = y_.size();
        for (size_t i = 0; i < size; i++) {
            ldiag_.push_back(i < ldiag.size() ? ldiag[i] : 0);
            udiag_.push_back(i < udiag.size() ? udiag[i] : 0);
            mdiag_.push_back(i < mdiag.size() ? mdiag[i] : 0);
        }
    };

    // initialize with size
    Tomas(vector<T> &ldiag,
          vector<T> &mdiag,
          vector<T> &udiag,
          vector<T> &y,
          size_t size) {
        for (int i = 0; i < size; i++) {
            ldiag_.push_back(i < ldiag.size() ? ldiag[i] : 0);
            mdiag_.push_back(i < mdiag.size() ? mdiag[i] : 0);
            udiag_.push_back(i < udiag.size() ? udiag[i] : 0);
            y_.push_back(i < y.size() ? y[i] : 0);
        }
    };

    // ------------------------------  destructor  --------------------------------
    ~Tomas() = default;

    // ----------------------------  member functions  ---------------------------

    inline const auto &lowerDiagonal() const { return ldiag_; };

    inline auto &lowerDiagonal() { return ldiag_; };

    inline const auto &midDiagonal() const { return mdiag_; };

    inline auto &midDiagonal() { return mdiag_; };

    inline const auto &upperDiagonal() const { return udiag_; };

    inline auto &upperDiagonal() { return udiag_; };

    const inline auto &yLeft() const { return yl_; };

    const inline auto &yRight() const { return yr_; };

    const inline auto &resuilt() const { return yr_; };

    const inline auto &y() const { return y_; };

    inline auto &y() { return y_; };


    void decompose() {
        // resize the matrix
        this->reSize();
        // d_i = c_i
        right_udiag_ = udiag_;
        // u_1 = b_1
        right_mdiag_[0] = mdiag_[0];
        // l_i = a_i / u_(i-1)
        // u_i = b_i - l_i*c_(i-1)
        for (int i = 1; i < y_.size(); i++) {
            left_ldiag_[i - 1] = ldiag_[i - 1] / right_mdiag_[i - 1];
            right_mdiag_[i] = mdiag_[i] - left_ldiag_[i - 1] * udiag_[i - 1];
        }
        // here we assume the left middle diagonal are all 1
        left_mdiag_ = left_mdiag_ + 1;
    };

    // build the left matrix
    vector<vector<double>> leftMatrix() {
        auto size = y_.size();
        vector<vector<double>> ans(size, vector<double>(size, 0));
        for (int i = 0; i < size; ++i) {
            if (i) ans[i][i - 1] = left_ldiag_[i - 1];
            ans[i][i] = left_mdiag_[i];
        }
        return ans;
    }

    // build the right matrix
    vector<vector<double>> rightMatrix() {
        auto size = y_.size();
        vector<vector<double>> ans(size, vector<double>(size, 0));
        for (int i = 0; i < size; ++i) {
            if (i < size - 1) ans[i][i + 1] = right_udiag_[i];
            ans[i][i] = right_mdiag_[i];
        }
        return ans;
    }

    void showMatrix(ostream &os = cout,
                    const string &filled = "-",
                    const int &precision = 1,
                    const string &gap = "\t") const {
        auto size = this->y_.size();
        for (int i = 0; i < size; i++) {
            // left space
            for (int spl = 0; spl < i - 1; spl++) {
                os << filled + gap;
            }
            // lower diagonal
            if (i) os << fixed << setprecision(precision) << this->ldiag_[i - 1] << gap;
            else os << "";
            // mid diagonal
            os << fixed << setprecision(precision) << this->mdiag_[i] << gap;
            // upper diagonal
            if (i < size - 1) os << fixed << setprecision(precision) << this->udiag_[i] << gap;
            else os << "  ";
            // right spaces
            for (int rsp = i + 2; rsp < size; rsp++) {
                os << filled + gap;
            }
            cout << gap << y_[i] << "\n";
            os << endl;
        }
    }

    // resize the matrix
    void reSize() {
        auto size = y_.size();
        // expand the size of the matrix
        resize(this->mdiag_, size);
        resize(this->ldiag_, size);
        resize(this->udiag_, size);
        resize(this->yr_, size);
        resize(this->yl_, size);
        // resize the decomposed matrix if it's computed
        resize(this->right_mdiag_, size);
        resize(this->left_mdiag_, size);
        resize(this->left_ldiag_, size);
        resize(this->right_udiag_, size);
    }


    void showDecomposedLeft(const string &filled = "-",
                            const int &precision = 3,
                            const string &gap = "\t") const {
        auto size = this->y_.size();
        for (int i = 0; i < size; i++) {
            // left space
            for (int spl = 0; spl < i - 1; spl++) {
                cout << filled + gap;
            }
            // lower diagonal
            if (i) cout << fixed << setprecision(precision) << this->left_ldiag_[i - 1] << gap;
            else cout << "";
            // mid diagonal
            cout << fixed << setprecision(precision) << this->left_mdiag_[i] << gap;
            // right spaces
            for (int rsp = i + 1; rsp < size; rsp++) {
                cout << filled << gap;
            }
            cout << "\n";
        }
        cout << "\n";
    }

    void showDecomposedRight(const string &filled = "-",
                             const int &precision = 3,
                             const string &gap = "\t") const {
        auto size = this->y_.size();
        for (int i = 0; i < size; i++) {
            // left space
            for (int spl = 0; spl < i; spl++) {
                cout << filled + gap;
            }
            // mid diagonal
            cout << fixed << setprecision(precision) << this->right_mdiag_[i] << gap;
            // upper diagonal
            if (i < size - 1) cout << fixed << setprecision(precision) << this->right_udiag_[i] << gap;
            else cout << gap;
            // right spaces
            for (int rsp = i + 2; rsp < size; rsp++) {
                cout << filled + gap;
            }
            cout << "\n";
        }
        cout << "\n";
    }

    // solve the equation
    void solve() {
        if (!prerequisite()) exit(1);
        decompose();
        solveEquation(left_mdiag_, left_ldiag_, 0);
        solveEquation(right_mdiag_, right_udiag_, 1);
    };


    // ----------------------------  overload  -------------------------------------
    Tomas<T> &operator=(const Tomas<T> &t) {
        this->mdiag_ = t.mdiag_;
        this->ldiag_ = t.ldiag_;
        this->udiag_ = t.udiag_;
        this->y_ = t.udiag_;
        this->yl_ = t.udiag_;
        this->yr_ = t.udiag_;
        this->right_mdiag_ = t.right_mdiag_;
        this->left_ldiag_ = t.left_ldiag_;
        this->left_mdiag_ = t.right_mdiag_;
        this->right_udiag_ = t.right_udiag_;
    }

};


#endif //CPP_PROGRAME_TOMAS_H
