# PAI - Private AI: 

## Practical Private AI based on Secure Multiparty Computing

PAI是一个基于MPC（安全多方计算）的隐私AI库，实现了从样本对齐到隐私模型训练的一站式解决方案。PAI通过定制化的训练协议，所有参与方代码可控，安全高效地进行联合建模。

### PAI Client - 底层MPC算法库

目前已经支持适用于多个参与方的 **纵向** 加密样本对齐、线性回归、逻辑回归和全连接神经网络算法。GBDT算法正在开发中。

### PAI Server - MPC任务调度和查询服务器

主节点运行Flask服务器，只需要一个Json POST请求，就可以创建MPC任务并且让各个节点开始执行任务。随时通过Http查询任务进展情况和测试集验证结果等信息。

### 文档

请访问 [PAI-Docs](https://zfscgy.github.io/PAI-doc/)