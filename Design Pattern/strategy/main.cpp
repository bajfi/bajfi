#include "classes.h"

int main(){
    classA A = classA();
    classB B = classB();
    classC C = classC();
    calculateMachine machineA = calculateMachine(&A);
    calculateMachine machineB = calculateMachine(&B);
    calculateMachine machineC = calculateMachine(&C);
    machineA.Calculate();
    machineB.Calculate();
    machineC.Calculate();

    machineA.set_machine(&B);
    machineA.Calculate();
    
    return 0;
}

