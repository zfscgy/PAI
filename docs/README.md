

# 基本模块

## Channel

一个用来通信的**抽象类**。

**`__init__`**：

* `self_id: int`：用来标识自己的id
* `n_clients: int`：表示总共有几个参与通信的节点
* `logger: Logger`：日志记录工具

**`send`**

* `receiver: int`：接收方的id

* `msg: ComputationMessage`：要发送的信息。必须是ComputationMessage类型的。

* `time_out: float`：超过该时间就放弃发送。注：只需要把消息放入接收方的缓存区即成功，并非要对应的`client`执行`receive`。

* **Return**：推荐在错误时返回None，反之返回收到的回应（ComputationMessage，确定消息被成功收到）。

  注意：该方法不开启一个新的线程，是阻塞的。

**`receive`**

* `sender: int`：发送方的id。

* `time_out`：超过该时间就停止接收。

* **Return**：推荐在错误时返回None，反之返回收到的消息（ComputationMessage，确定消息被成功收到）。

  注意：该方法不开启新线程，是阻塞性的。

这个类的方法都是阻塞性的，且有一个发送必须有一个接收，否则缓存区会被占据；缓存区只能存放一个消息。因此所有的通信协议必须严格制定。

*这个类是一个抽象类，方法并没有进行实现。*

## BaseClient

表示基本的客户端类，在`Channel`上封装了一层。可以调用不同的`Channel`的具体实现。

**`__init__`**

* `channel: Channel`：

**`send_msg`**：直接调用了`Channel`的对应方法。

**`receive_msg`**：直接调用了`Channel`的对应方法。

**`send_check_msg`**

* `receiver: int`：接收方
* `msg: ComputationMessage`：要发送的信息。

**`receive_check_msg`**

* `sender: int `：发送方
* `header: MessageType`：指定要接收的`MessageType`，也可以是`MessageType`的列表
* `time_out: float`
* **Return**：如果未能接收到指定类型的消息，会报错。返回接收到的消息。



## MPC计算神经网络

在MPC计算神经网络的过程中，需要有3个参与方：

* DataClient: 拥有数据
* LabelClient: 拥有标签
* MainClient: 拥有一部分模型

基本的运算流程如下：

| Stage         | data_client                                    | Main Client(Server)                    | label_client send        |
| ------------- | ---------------------------------------------- | -------------------------------------- | ------------------------ |
| Preparing     |                                                | TRAIN_CONFIG **→ All**                 |                          |
|               | CLIENT_READY **→Server**                       |                                        | **Server←** CLIENT_READY |
| Training loop |                                                | NEXT_TRAIN_ROUND **→ All**             |                          |
|               | *doing hidden layer computations....*          |                                        |                          |
|               | MUL_OUT_SHARE **→Server**                      |                                        |                          |
|               |                                                | *calculating server outputs*           |                          |
|               |                                                | PRED_LABEL <br />**→ Label client**    |                          |
|               |                                                |                                        | *calculating gradients*  |
|               |                                                |                                        | **Server←** PRED_GRAD    |
|               |                                                | *calculating server gradients*         |                          |
|               |                                                | CLIENT_OUT_GRAD<br />**Data clients←** |                          |
|               | *exchanging gradients with other data clients* |                                        |                          |
|               | CLIENT_ROUND_OVER **→Server**                  |                                        |                          |

