#include "interface.h"


// A component
class classA:public classBase
{
private:
    /* data */
public:
    classA(/* args */) = default;
    ~classA() override = default;

    void calculate() noexcept override{
        cout << "Calculate with A method" << endl;
    }
};


// B component
class classB : public classBase
{
private:
    /* data */
public:
    classB() = default;
    ~classB() override  = default;

    void calculate() noexcept override{
        cout << "Calculate with B method" << endl;
    }
};

// C component
class classC : public classBase
{
private:
    /* data */
public:
    classC() = default;
    ~classC() override = default;

    void calculate() noexcept override{
        cout << "Calculate with C method" << endl;
    }
};



// The context part
class calculateMachine
{
private:
    /* data */
    classBase* _base{};
public:
    calculateMachine() = default;

    explicit calculateMachine(classBase* base):_base(base){};

    explicit calculateMachine(classBase& base):_base(&base){};

    virtual ~calculateMachine() = default;

    virtual void Calculate() noexcept{
        this->_base->calculate();
    };

    void set_machine(classBase* base){
        this->_base = base;
    }

    void set_machine(classBase& base){
        this->_base = &base;
    }
};