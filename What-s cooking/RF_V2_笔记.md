与上一个版本的区别有以下几点：

1、处理数据使用pandas库，相比于之前处理是使用Python基本数据结构，要方便；

2、对于分类特征的编码，上一个版本是自己写的one-hot，这一次使用的是CounterVector()，bag_of_word；

3、没有对分类标签进行数字型转化，还是采用字符类型，RF可以直接使用字符型数据当作标签；

4、在使用决策树时候，用到了包外测试，使用了从训练集中划分的一部分作为验证集。from sklearn.model_selection import train_test_split。

最后，accuracy从0.71提高到了0.75。
