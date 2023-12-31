##### 模板方法（template method）是一个定义在父类的方法，负责处理流程、算法的不变部分。模板方法会调用多个定义在父类的其他工具方法（helper method），而这些方法是算法的可变部分，有可能只是抽象方法并没有实现。模板方法仅决定这些抽象方法的执行顺序，这些抽象方法由子类负责实现，并且子类不允许覆盖模板方法（即不能重写处理流程）。

![](https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Template_Method_UML.svg/300px-Template_Method_UML.svg.png)