
#ifndef OBSERVER_SUBSCRIBECLASSES_H
#define OBSERVER_SUBSCRIBECLASSES_H

#include <unordered_set>
#include "subscriberBase.h"

class subscribeA:public subscribeBase{
public:
    virtual void operations() noexcept{
        cout << "do something about A subscriber" << "\n";
    };

    void notify() noexcept override{
        cout << "subscribe A has been notified !" << "\n";
    };
};

class subscribeB:public subscribeBase{
public:
    virtual void operations(){
        cout << "do something about B subscriber" << "\n";
    };

    void notify() noexcept override{
        cout << "subscribe B has been notified !" << "\n";
    };
};

class subscribeC:public subscribeBase{
public:
    virtual void operations(){
        cout << "do something about C subscriber" << "\n";
    };

    void notify() noexcept override{
        cout << "subscribe C has been notified !" << "\n";
    };
};


// Mail System
class MailSystem{
private:
    unordered_set<subscribeBase*> _subscribers{};

public:
    MailSystem()=default;
    explicit MailSystem(unordered_set<subscribeBase*>& subscribers):_subscribers(subscribers){};
    ~MailSystem()=default;

    inline void addSubscribe(subscribeBase* subscriber) noexcept {
        _subscribers.insert(subscriber);
    }

    inline void addSubscribe(subscribeBase& subscriber) noexcept {
        _subscribers.insert(&subscriber);
    }

    inline void removeSubscribe(subscribeBase* subscriber) noexcept {
        _subscribers.erase(subscriber);
    }

    inline void removeSubscribe(subscribeBase& subscriber) noexcept {
        _subscribers.erase(&subscriber);
    }

    void notify() noexcept {
        for(auto subscriber:_subscribers){
            subscriber->notify();
        }
    }

};

#endif //OBSERVER_SUBSCRIBECLASSES_H
