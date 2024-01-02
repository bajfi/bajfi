
#ifndef DECORATOR_DECORATOR_H
#define DECORATOR_DECORATOR_H

#include <iostream>
#include "Tool.h"

using namespace std;

class decorator:public Tool{
private:
    Tool* _tool{};
public:
    explicit decorator(Tool* tool):_tool(tool){}
    ~decorator() override = default;
    void perform() override {
        _tool->perform();
    };
};

class decWalk:public decorator{
public:
    explicit decWalk(Tool* tool): decorator(tool){}
    ~decWalk() override=default;
    void perform() noexcept override{
        decorator::perform();
        cout << "Walking ..." << "\n";
    }
};

class decJump:public decorator{
public:
    explicit decJump(Tool* tool): decorator(tool){}
    ~decJump() override = default;
    void perform() noexcept override{
        decorator::perform();
        cout << "jumping ... " << "\n";
    }
};

class decSpeak:public decorator{
public:
    explicit decSpeak(Tool* tool): decorator(tool){}
    ~decSpeak() override = default;
    void perform() noexcept override{
        decorator::perform();
        cout << "speaking ... " << "\n";
    }
};

#endif //DECORATOR_DECORATOR_H
