
#ifndef BRIDGE_DRAWMETHOD_H
#define BRIDGE_DRAWMETHOD_H

#include <iostream>

using namespace std;


class DrawMethod_Imp{
public:
    virtual ~DrawMethod_Imp()=default;
    virtual void drawMethod()=0;
};

class DrawMethodA:public DrawMethod_Imp{
public:
    ~DrawMethodA() override =default;
    void drawMethod() noexcept override{
        cout << " with A method" << "\n";
    }
};

class DrawMethodB:public DrawMethod_Imp{
public:
    ~DrawMethodB() override =default;
    void drawMethod() noexcept override{
        cout << " with B method" << "\n";
    }
};

#endif //BRIDGE_DRAWMETHOD_H
