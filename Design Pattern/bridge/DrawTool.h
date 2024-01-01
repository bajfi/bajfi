
#ifndef BRIDGE_DRAWTOOL_H
#define BRIDGE_DRAWTOOL_H

#include <iostream>
#include "DrawMethod.h"

using namespace std;

// here DrawToolBase means Shape
class DrawToolBase{
public:
    DrawMethod_Imp* _drawMethod_imp{};
    virtual ~DrawToolBase()=default;
    virtual void Draw()=0;

    void setMethod(DrawMethod_Imp* drawMethodImp){
        this->_drawMethod_imp = drawMethodImp;
    }

    void setMethod(DrawMethod_Imp& drawMethodImp){
        this->_drawMethod_imp = &drawMethodImp;
    }
};

// ================================================================
// draw a circle
class CircleDrawTool:public DrawToolBase{
public:
    explicit CircleDrawTool(DrawMethod_Imp* drawMethodImp){
        this->_drawMethod_imp = drawMethodImp;
    };
    explicit CircleDrawTool(DrawMethod_Imp& drawMethodImp){
        this->_drawMethod_imp = &drawMethodImp;
    }
    ~CircleDrawTool() override= default;

    void Draw() noexcept override{
        cout << "Draw " << typeid(*this).name();
        this->_drawMethod_imp->drawMethod();
    }
};

// draw a triangle
class TriangleDrawTool:public DrawToolBase{
public:
    explicit TriangleDrawTool(DrawMethod_Imp* drawMethodImp){
        this->_drawMethod_imp = drawMethodImp;
    };
    explicit TriangleDrawTool(DrawMethod_Imp& drawMethodImp){
        this->_drawMethod_imp = &drawMethodImp;
    }
    ~TriangleDrawTool() override= default;

    void Draw() noexcept override{
        cout << "Draw " << typeid(*this).name();
        this->_drawMethod_imp->drawMethod();
    }
};

#endif //BRIDGE_DRAWTOOL_H
