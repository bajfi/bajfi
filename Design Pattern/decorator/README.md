#### Decorator Design Pattern

**修饰模式，是面向对象程式领域中，一种动态地往一个类别中添加新的行为的设计模式。就功能而言，修饰模式相比生成子类别更为灵活，这样可以给某个对象而不是整个类别添加一些功能**

（我们可以将`decorator`理解为不同功能的积木，我们可以将不同功能的`decorator`进行组装，已达到自己的所需。）

![](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Decorator_UML_class_diagram.svg/400px-Decorator_UML_class_diagram.svg.png)

这里我们以实现玩具`Tool`为例，其中：
* `Tool`为玩具的基类,具有特定的方法，其中`perform()`为需要扩充的功能；
* `decorator`为方法类的基类,同时聚合和继承`Tool`方法类，方便后续扩充接口功能，这里的扩充只是对某些特定的功能进行装饰，因此需要保持函数接口相同；