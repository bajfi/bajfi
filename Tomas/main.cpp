
#include "Tomas.cpp"


int main() {
    vector<double> a(4, -1);
    vector<double> b(5, 3);
    vector<double> c(5, 2);
    vector<double> y{7, 11, 15, 9};
    Tomas<double> t(a, b, c, y);
    t.showMatrix();
    t.decompose();
    auto l = t.leftMatrix();
    auto r = t.rightMatrix();
    auto m = l * r;
    t.solve();
    auto resultt = t.resuilt();
    cout << m * resultt;

}