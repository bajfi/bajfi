#include "SOR.cpp"
#include <ctime>
#include <random>

vector<double> generate_random(int N, int low, int high)
{
    vector<double> A(N);
    for (int i = 0; i < N; ++i) {
        A[i] = double(rand()) / RAND_MAX * (high - low) + low;
    }
    return A;
}

vector<vector<double>> generate_random(int M, int N, int low, int high)
{
    vector<vector<double>> A(M, vector<double>(N));
    for (int i = 0; i < M; ++i) {
        for (int j = i; j < N; ++j) {
            A[i][j] = double(rand()) / RAND_MAX * (high - low) + low;
            A[j][i] = A[i][j];      // symmetry
        }
    }
    for (int i = 0; i < M; ++i) {
        // make sure diagonal domain
        A[i][i] = N * high;
    }
    return A;
}

int main()
{
    srand((unsigned) time(nullptr));
    int N = 10;
    auto A = generate_random(N, N, -10, 10);
    auto b = generate_random(N, -100, 100);
    auto x = generate_random(N, -1, 1);
    SOR sor = SOR(A, b, x);
    cout << sor;
    sor.solve();
    cout << "error: \n" << sor.error();
    return 0;
}
