* # 利用追赶法求解三对角矩阵

* ## How to use

**For matrix like** :
| a1 | c1 | 0  | 0  | 0  |
|----|----|----|----|----|
| b1 | a2 | c2 | 0  | 0  |
| 0  | b2 | a3 | c3 | 0  |
| 0  | 0  | b3 | a4 | c4 |
| 0  | 0  | 0  | b4 | a5 |

### Code example:
```
    // initialize
    vector<double> a(4, -1);
    vector<double> b(5, 3);
    vector<double> c(5, 2);
    vector<double> y{7,11,15,9};
    Tomas<double> t(a, b, c, y);

    // show the function
    cout << t;      // which is same as " t.showMatrix(); "

    // decompose the matrix
    t.decompose();

    // check the decomposed matrix
    t.showDecomposedLeft()      // and t.showDecomposedRight()

    // or we can get the left(right) decomposed matrix
    auto left = t.leftMatrix();
    auto right = t.rightMatrix();
    cout << left;   // and show the matrix
    cout << right;

    // use t.solve() to solve the function
    t.solve()

    // use t.result() to show the solution
    auto result =  t.result();
    cout << result;

    // we can also verify the result
    // by check if left*right*result is equal to y or not
    auto m = left*right;
    cout << m * result;

```