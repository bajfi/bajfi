
#ifndef TEMPLATE_METHOD_TEMPLATECLASS_H
#define TEMPLATE_METHOD_TEMPLATECLASS_H

// the template class keeps the works flow unchanged
class templateClass{
public:
    templateClass()=default;
    virtual ~templateClass() = default;

    virtual void step1()=0;
    virtual void step2()=0;
    virtual void step3()=0;
    virtual void step4()=0;

    void execute() noexcept{
        step1();
        step2();
        step3();
        step4();
    }
};

#endif //TEMPLATE_METHOD_TEMPLATECLASS_H
