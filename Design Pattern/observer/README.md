**观察者模式是软件设计模式的一种。在此种模式中，一个目标对象管理所有相依于它的观察者对象，并且在它本身的状态改变时主动发出通知。这通常透过呼叫各观察者所提供的方法来实现。此种模式通常被用在即时事件处理系统。**

![](https://upload.wikimedia.org/wikipedia/commons/e/e2/Observer-pattern-class-diagram.png)

这里我们一个邮件订阅系统，其中：
* `MailSystem`为系统，
    * `addSubscribe()`允许添加订阅者；
    * `removeSubscive()`允许移除订阅者；
    * `notify()`对每一个订阅者发送通知
* `subscribeClasses`中包含多种类型的订阅对象,不同类型对象的`notify`方式也有所不同