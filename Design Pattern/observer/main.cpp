#include "subscribeClasses.h"

int main() {
    subscribeA subA = subscribeA();
    subscribeB subB = subscribeB();
    subscribeC subC = subscribeC();
    MailSystem sys = MailSystem();

    subA.operations();
    subB.operations();
    subC.operations();

    cout << "\n";

    sys.addSubscribe(subA);
    sys.addSubscribe(subB);
    sys.addSubscribe(subC);

    sys.notify();
    cout << "\n";

    sys.removeSubscribe(subA);
    sys.notify();

    return 0;
}
