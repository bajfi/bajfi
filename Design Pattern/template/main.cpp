#include <iostream>
#include "workflow.h"

int main() {
    workflow1 wf1 = workflow1();
    workflow2 wf2 = workflow2();
    Application app1 = Application(&wf1);
    Application app2 = Application(&wf2);

    app1.execute();
    app2.execute();

    app1.setWorkflow(&wf2);
    app1.execute();

    return 0;
}
