#include "decorator.h"

int main() {
    ToolA toolA = ToolA();
    decWalk walk = decWalk(&toolA);
    decJump jump = decJump(&walk);
    decSpeak speak = decSpeak(&jump);

    toolA.perform();
    walk.perform();
    jump.perform();
    speak.perform();
    return 0;
}
