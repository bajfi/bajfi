#include <iostream>

using namespace std;

class classBase
{
private:
    /* data */
public:
    classBase(/* args */) = default;
    virtual ~classBase(){};

    virtual void calculate() = 0;
};


