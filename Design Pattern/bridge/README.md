#### Bridge Design Pattern

**桥接模式是软件设计模式中最复杂的模式之一，它把事物对象和其具体行为、具体特征分离开来，使它们可以各自独立的变化。事物对象仅是一个抽象的概念。**

![](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/Bridge_UML_class_diagram.svg/600px-Bridge_UML_class_diagram.svg.png)

这里我们设计一个绘图工具，其中：
`DrawToolBase`为绘图工具基类，聚合绘制方法基类，这里指代图形的形状；
`DrawTool`为实例化的绘图基类，这里指代具体的形状，如（Circle,Triangle）；
`DrawMethod_Imp`为绘制方法基类；
`DrawMethod`为具体的绘制方法，同一种形状可能有多种绘制方法。

**由于图形的形状种类和绘制方式都具有可变性，因此将其设计为抽象类，方便扩展。**

在重构之前，假设系统存在**n**个`DrawTool`，**m**个`DrawMethod`，对于每一个`DrawTool`即shape而言，都需要实现$ {\textstyle \sum_{1}^{m}C_{m}^{i} } \approx m^{2} $个子类，总共需要实现$nm^2$个子类；

而重构之后，只需要对每一个方法和形状单独实现即可，即只需要实现约$(m+n)$个子类即可。