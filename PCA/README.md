# PCA

降维技术
====

主成分分析
----
由数据本身选择新坐标系，按照规则不断地选择新坐标轴，最后发现大部分方差都包含在最前面的几个新坐标轴中，忽略剩下的坐标轴，即实现降维处理  
__优点__： 降低数据的复杂性，识别最重要的多个特征  
__缺点__： 不一定需要，且可能损失有用信息  
__适用数据类型__： 数值型数据  

因子分析
----
假设观察数据是隐变量和某些噪声的线性组合，找到隐变量实现数据的降维

独立成分分析
----
假设数据是从N个数据源生成的，假设数据为多个数据源的混合观察结果，这些数据源之间在统计上是相互独立的，而在PCA中只假设数据是不相关的，同因子分析一样，如果数据源的数目少于观察数据的数目，则可以实现降维过程。
