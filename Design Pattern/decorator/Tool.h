
#ifndef DECORATOR_TOOL_H
#define DECORATOR_TOOL_H

#include <iostream>
#include <typeinfo>

using namespace std;

class Tool{
public:
    virtual ~Tool()= default;
    virtual void perform() = 0;
};

class ToolA:public Tool{
public:
    ~ToolA() override=default;
    void perform() noexcept override{
        cout << typeid(*this).name() << " performing \n";
    }
    virtual void Afunc() noexcept{
        cout << "performing A functions" << "\n";
    }
};

#endif //DECORATOR_TOOL_H
