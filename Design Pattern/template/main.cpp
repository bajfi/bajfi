#include <iostream>
#include "workflow.h"

int main() {
    auto wf1 = workflow1();
    auto wf2 = workflow2();
//    auto* wf1 = new workflow1();
//    auto* wf2 = new workflow2();
    auto app1 = Application(&wf1);
    auto app2 = Application(&wf2);
//    auto* app1 = new Application(&wf1);
//    auto* app2 = new Application(&wf2);

    app1.execute();
    app2.execute();

    app1.setWorkflow(&wf2);
    app1.execute();
    
//    delete wf1;
//    delete wf2;
//    delete app1;
//    delete app2;

    return 0;
}
