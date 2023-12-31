#include <iostream>

using namespace std;

class classBase
{
private:
    /* data */
public:
    classBase(/* args */) = default;
    virtual ~classBase(){};

    // caculate method is changable
    virtual void calculate() = 0;
};


