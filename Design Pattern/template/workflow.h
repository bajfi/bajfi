
#ifndef TEMPLATE_METHOD_WORKFLOW_H
#define TEMPLATE_METHOD_WORKFLOW_H

#include <iostream>
#include "templateClass.h"

using namespace std;

class workflow1:public templateClass{
    void step1() override{
        cout << "workflow1 finish step1" << "\n";
    }

    void step2() override{
        cout << "workflow1 finish step2" << "\n";
    }

    void step3() override{
        cout << "workflow1 finish step3" << "\n";
    }

    void step4() override{
        cout << "workflow1 finish step4" << endl;
    }
};

class workflow2:public templateClass{
    void step1() override{
        cout << "workflow2 finish step1" << "\n";
    }

    void step2() override{
        cout << "workflow2 finish step2" << "\n";
    }

    void step3() override{
        cout << "workflow2 finish step3" << "\n";
    }

    void step4() override{
        cout << "workflow2 finish step4" << endl;
    }
};


class Application{
private:
    templateClass* workflow{};
public:
    explicit Application(templateClass* temp):workflow(temp){};

    ~Application(){
        if(workflow != nullptr){
            delete workflow;
            workflow = nullptr;
        }
    }

    virtual void execute() noexcept{
        workflow->execute();
    }

    void setWorkflow(templateClass* tmp) noexcept{
        if(workflow!= nullptr){
            delete workflow;
            workflow = nullptr;
        }
        workflow = tmp;
    }
};

#endif //TEMPLATE_METHOD_WORKFLOW_H
