# AdaBoost
一种集成学习方法

bagging
----
自聚汇聚法，bootstrap抽样，每个分类器权重一样

boosting
----
集中关注被已有分类器错分的那些数据来获得新的分类器
分类权重不相等，权重依赖于上一轮迭代中的成功度

选择分类器
----
选择的都是弱分类器，如果有强分类器在里面，其所占的alpha会很大，其他的分离器都没用了，就违背了adaboost方法的初衷

__优点__： 泛化错误率低，易编码，可以应用在大部分分类器上，无参数调整  
__缺点__： 对离群点敏感  
__适用数据类型__： 数值型和标称型数据  
