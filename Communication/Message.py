from enum import Enum


class MessageType(Enum):
    NULL = 0

    # 以下指令主要用于Secret-sharing矩阵乘法，主要场景为一个客户端提供数据，另一个提供参数，求其变换后的数据
    # 比如：一个客户端的数据shape = (batch_size, dim)，另一个客户端拥有对应的参数 (dim, out_dim)
    # 该方法可以计算出这两个矩阵的乘积，而客户端的自己的数据不需要发送出去。



    SET_TRIPLET = 11
    """
    Send a beaver triple request to triple provider
    The data should be:
    `Tuple(target_client:int, shape_sender:Tuple(int, int), shape_target:Tuple(int, int))`
    """

    TRIPLE_ARRAY = 12
    """
    Used for triple provider, for send the triple to the two parties
    """

    TRAIN_CONFIG = 14
    """
    Configurations for training, should be sent by the main client before start traing,
    including: data_dimensions, batch_sizes
    """

    CLIENT_READY = 15
    """
    """

    NEXT_TRAIN_ROUND = 17
    """
    Client start next training round
    the data usually is None
    If the data is "Test", then this round should be performed in test mode
    """

    PARA_GRAD = 18

    TRAINING_STOP = 19
    # Those is for shared matrix multiplication

    MUL_DATA_SHARE = 20
    """
    Used by `DataClient`, to send shares of its data to the other party
    """

    MUL_OwnVal_SHARE = 21
    """
    Used by `DataClient`, send its A - U to the other party 
    (A is its share of its own matrix, and U is its share of its triple)
    """

    MUL_OtherVal_SHARE = 22
    """
    Used by `DataClient`, send its B - V to the other party 
    (B is its share of the other party's matrix, and V is its share of the other party's triple)
    """

    MUL_OUT_SHARE = 24

    PRED_LABEL = 28
    PRED_GRAD = 29
    CLIENT_OUT_GRAD = 30
    CLIENT_PARA_UPDATE = 31
    CLIENT_ROUND_OVER = 32

    # Those is for message responses
    RECEIVED_ERR = 98
    RECEIVED_OK = 99


class ReceiveERRType(Enum):
    BUFFER_OCCUPIED = 1
    UNRECOGNIZED_SENDER = 2
    CONNECTION_FAILED = 3


class ComputationMessage:
    """
    Message class
    """
    def __init__(self, header: MessageType, data):
        """
        :param header:  Header of the message, specify the type of message
        :param data: Message data

        Initialize a Message
        """
        self.header = header
        self.data = data

    def __str__(self):
        return "header:" + self.header.__str__() + "\ndata:" + self.data.__str__() + "\n"
