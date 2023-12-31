
#ifndef OBSERVER_SUBSCRIBERBASE_H
#define OBSERVER_SUBSCRIBERBASE_H

#include <iostream>

using namespace std;

class subscribeBase{
public:
    subscribeBase()=default;
    virtual void notify()=0;
    virtual ~subscribeBase()=default;
};

#endif //OBSERVER_SUBSCRIBERBASE_H
