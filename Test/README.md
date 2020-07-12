## 单机示例：双方提供数据，另一方提供标签，训练违约预测模型

### 建立IP地址映射表

```python
ip_dict = {
    0: "127.0.0.1:19001",
    1: "127.0.0.1:19002",
    2: "127.0.0.1:19003",
    3: "127.0.0.1:19004",
    4: "127.0.0.1:19005"
}
```

IP地址映射表把IP映射成数字`client_id`，提供给`Channel`，而后续的`BaseClient` 类的通信只需要指定`client_id`，不需要使用IP地址。

### Channel 初始化

```python
channel0 = Peer(0, "[::]:19001", 10, ip_dict, 3, logger=Logger(prefix="Channel0:"))
channel1 = Peer(1, "[::]:19002", 10, ip_dict, 3, logger=Logger(prefix="Channel1:", level=1))
channel2 = Peer(2, "[::]:19003", 10, ip_dict, 3, logger=Logger(prefix="Channel2:"))
channel3 = Peer(3, "[::]:19004", 10, ip_dict, 3, logger=Logger(prefix="Channel3:"))
channel4 = Peer(4, "[::]:19005", 10, ip_dict, 3, logger=Logger(prefix="Channel4:"))
```

初始化5个`Channel`。比如：`channel1 = Peer(1, "[::]:19002", 10, ip_dict, 3, logger=Logger(prefix="Channel1:", level=1))`，第一个参数 `1` 表示其`client_id`，第二个参数表示监听19002端口（所有IP），第三个参数表示`max_workers`，是`GRPC`中使用的。第四个参数是IP地址映射表。注意1、2、4这三个参数必须匹配，`ip_dict[client_id]`中对应的端口号必须是第二个参数中的端口号。第五个参数表示最大延时，即：如果当前`Channel`的某个缓冲区是被占用的，最多要等待的时间。`logger` 指日志记录器。

### 初始化主客户端

`main_client = MainTFClient(channel0, [2, 3], 4, logger=Logger(prefix="Main client:"))`

主客户端，即所谓的"服务器"。客户端初始化的第一个参数都是`Channel`，客户端需要依托于该 `Channel` 和其他客户端进行通信。

主客户端的第二个参数 `[2, 3]` 表示提供数据的客户端的`client_id`是2,3。第三个参数 `4` 表示提供标签的客户端的`client_id`。`logger`表示日志记录器。

### 初始化三元组客户端

```python
triplets_provider = TripletsProvider(channel1, logger=Logger(prefix="Triplet provider:"))
```

为了实现加法分享的矩阵乘法，需要有中立的第三方提供三元组。三元组客户端通过监听所有其他用户，来进行三元组生成。比如`client_id`为1,2的两个用户之间要进行矩阵乘法，其中1用户的矩阵大小为$10 \times 20$，2用户的矩阵大小为 $20\times 30$：1用户发送消息`(1, 2, (10, 20), (20, 30))` 给三元组客户端，此时三元组客户端将该消息存储在内部缓冲区（看做一个二维数组）的 `[1, 2]` 位置，直到2用户发送消息`(2, 1, (20, 30), (10, 20))`给三元组客户端，此时三元组客户端会查看该消息是否与内部缓冲区`[1, 2]`位置中的消息相匹配，如果匹配则进行三元组产生过程，并把对应的三元组分别发送给1、2两方。

### 初始化数据客户端

```python
data_client0 = DataClient(channel2,
                          CSVDataLoader("Test/TestDataset/credit_default.csv", 
                                        list(range(40000)), list(range(30))),
                          CSVDataLoader("Test/TestDataset/credit_default.csv", 
                                        list(range(40000, 50000)), list(range(30))),
                          server_id=0, triplets_id=1, other_data_clients=[3],
                          logger=Logger(prefix="Data client 0:"))
data_client1 = DataClient(channel3,
                          CSVDataLoader("Test/TestDataset/credit_default.csv", 
                                        list(range(40000)), list(range(30, 72))),
                          CSVDataLoader("Test/TestDataset/credit_default.csv", 
                                        list(range(40000, 50000)),
                                        list(range(30, 72))),
                          server_id=0, triplets_id=1, other_data_clients=[2],
                          logger=Logger(prefix="Data client 1:"))
```

数据客户端的第一个参数也是`Channel`，第二个参数是训练数据Loader，用于读取训练数据集，第三个参数是测试数据Loader，用于读取测试数据集。它们都属于`CSVDataLoader`，这个类初始化的第一个参数是csv文件的路径，第二、三个参数是数组，分别指定需要载入的行和列。比如`list(range(40000))`，表示只载入第0-39999行。

此外，还需要指定服务器（主客户端）`server_id`，三元组客户端`triplets_id`，其他数矩客户端`other_data_clients`。

### 初始化标签客户端

```python
def auc(y_true, y_pred):
    return roc_auc_score(y_true[:, 0], y_pred[:, 0])
label_client = LabelClient(channel4,
                           CSVDataLoader("Test/TestDataset/credit_default.csv", 
                                         list(range(40000)), list(range(72, 73))),
                           CSVDataLoader("Test/TestDataset/credit_default.csv", 
                                         list(range(40000, 50000)),list(range(72, 73))),
                           server_id=0, metric_func=auc, logger=Logger(prefix="Lable client:"))
```

标签客户端同样需要指定训练和测试数据Loader，只是载入的是标签。同时需要指定服务器（主客户端）`server_id`，以及一个效果度量函数`metric_func`，在这里是`AUC(Area under roc curve)`。每一轮训练的时候`Logger`会打印度量函数得到的值。

### 三元组客户端启动

```python
triplets_provider.start_listening()
```

由于这里是在一个Python脚本中运行，因此需要用多线程的方式。注意三元组客户端本身的`start_listening`就是启动多个线程，不是阻塞的。而其他的客户端的`start_train`是阻塞的，因此需要用多线程的方式启动。

### 设置训练配置信息

```python
config = {
    "client_dims": {2: 30, 3: 42}, # 各个数矩客户端的维度
    "out_dim": 1, # 各个客户端的输出维度
    "batch_size": 256,
    "test_per_batch": 100, # 每100轮要进行一次测试
    "test_batch_size": None, # None表示整个测试集都测试
    "learning_rate": 0.01,
    "sync_info": { # 用于数据Loader同步，使得各个客户端每个batch加载的样本是一样的。
        "seed": 8888 # 数据Loader的随机种子
    }
}
```

这些配置信息将会由主客户端发送给各个数据客户端。

### 主客户端计算单元初始化

由于我们认为，主客户端的具体网络结构与训练配置信息并不一样，因此这里没有整合到`config`里面去，而是额外的一个方法。

```python
main_client.__build_mlp_network(1, [])
```

这里表示构建一个输入为1，不包含其它层的MLP网络，即仅仅对输入做了一个sigmoid。因为当前场景是逻辑回归。

### 各个客户端启动

```python
data_client0_th = threading.Thread(target=data_client0.start_train)
data_client1_th = threading.Thread(target=data_client1.start_train)
label_client_th = threading.Thread(target=label_client.start_train)
main_client_start_th = threading.Thread(
    target=main_client.start_train,
    args=(config,)
)
data_client0_th.start()
data_client1_th.start()
label_client_th.start()
main_client_start_th.start()
triplets_provider.start_listening()
```

### 启动之后的日志信息示例（忽略tensorflow warnings）

>Channel0:[INFO] [2020-06-24 10:01:25.172879]  Peer id 0 started.
>Channel2:[INFO] [2020-06-24 10:01:25.172879]  Peer id 2 started.
>Channel3:[INFO] [2020-06-24 10:01:25.181591]  Peer id 3 started.
>Channel4:[INFO] [2020-06-24 10:01:25.181591]  Peer id 4 started.
>Data client 0:[INFO] [2020-06-24 10:01:38.714328]  Client initialized
>Data client 1:[INFO] [2020-06-24 10:01:51.757129]  Client initialized
>Lable client:[INFO] [2020-06-24 10:02:05.628395]  Client initialized
>
>Data client 0:[INFO] [2020-06-24 10:02:06.648144]  Client started, waiting for server config message with time out 100.00
>Data client 1:[INFO] [2020-06-24 10:02:06.648144]  Client started, waiting for server config message with time out 100.00
>Lable client:[INFO] [2020-06-24 10:02:06.648144]  Client started, waiting for server config message with time out 100.00
>====== Stop the triplet provider, the training should be auto exited =========
>Lable client:[INFO] [2020-06-24 10:02:06.725037]  Received train config message: {'client_dims': {2: 30, 3: 42}, 'out_dim': 1, 'batch_size': 256, 'test_per_batch': 100, 'test_batch_size': None, 'learning_rate': 0.01, 'sync_info': {'seed': 8888}}
>Data client 0:[INFO] [2020-06-24 10:02:06.725037]  Received train conifg message: {'client_dims': {2: 30, 3: 42}, 'out_dim': 1, 'batch_size': 256, 'test_per_batch': 100, 'test_batch_size': None, 'learning_rate': 0.01, 'sync_info': {'seed': 8888}}
>Data client 1:[INFO] [2020-06-24 10:02:06.729391]  Received train conifg message: {'client_dims': {2: 30, 3: 42}, 'out_dim': 1, 'batch_size': 256, 'test_per_batch': 100, 'test_batch_size': None, 'learning_rate': 0.01, 'sync_info': {'seed': 8888}}
>Lable client:[INFO] [2020-06-24 10:02:06.760185]  Test Round:
>Data client 0:[INFO] [2020-06-24 10:02:06.760185]  Test Round:
>Data client 1:[INFO] [2020-06-24 10:02:06.762226]  Test Round:
>Data client 1:[INFO] [2020-06-24 10:02:07.468174]  Train round 1 finished
>Data client 0:[INFO] [2020-06-24 10:02:07.468174]  Train round 1 finished
>Lable client:[INFO] [2020-06-24 10:02:07.532362]  Current batch loss: 0.2512, metric value: 0.5627
>Lable client:[INFO] [2020-06-24 10:02:07.555536]  Train round 1 finished
>Main client:[INFO] [2020-06-24 10:02:07.579851]  Train round 1 finished
>Lable client:[INFO] [2020-06-24 10:02:07.893228]  Current batch loss: 0.2510, metric value: 0.5238
>Lable client:[INFO] [2020-06-24 10:02:07.904370]  Train round 2 finished
>Data client 0:[INFO] [2020-06-24 10:02:08.329677]  Train round 2 finished
>Data client 1:[INFO] [2020-06-24 10:02:08.329677]  Train round 2 finished
>Main client:[INFO] [2020-06-24 10:02:08.351879]  Train round 2 finished
>Lable client:[INFO] [2020-06-24 10:02:08.755960]  Current batch loss: 0.2503, metric value: 0.5705
>Lable client:[INFO] [2020-06-24 10:02:08.779284]  Train round 3 finished
>Data client 1:[INFO] [2020-06-24 10:02:08.959662]  Train round 3 finished
>Data client 0:[INFO] [2020-06-24 10:02:08.971549]  Train round 3 finished
>Main client:[INFO] [2020-06-24 10:02:08.994874]  Train round 3 finished

