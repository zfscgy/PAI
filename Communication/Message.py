from enum import Enum


class _AutoIndexer:
    def __init__(self):
        self._index = 0

    def auto(self):
        self._index += 1
        return self._index


indexer = _AutoIndexer()


class MessageType(Enum):
    NULL = indexer.auto()

    Common_Stop = indexer.auto()

    Triplet_Set = indexer.auto()
    Triplet_Array = indexer.auto()

    SharedNN_RandomSeed = indexer.auto()
    SharedNN_ClientDim = indexer.auto()
    SharedNN_TrainConfig = indexer.auto()
    SharedNN_FeatureClientOut = indexer.auto()
    SharedNN_MainClientOut = indexer.auto()
    SharedNN_MainClientGradLoss = indexer.auto()
    SharedNN_FeatureClientGrad = indexer.auto()
    SharedNN_FeatureClientParaGrad = indexer.auto()

    CLIENT_READY = indexer.auto()
    """
    """

    NEXT_TRAIN_ROUND = indexer.auto()
    """
    Client start next training round
    the data usually is None
    If the data is "Test", then this round should be performed in test mode
    """

    PARA_GRAD = indexer.auto()

    TRAINING_STOP = indexer.auto()
    # Those is for shared matrix multiplication

    MUL_DATA_SHARE = indexer.auto()
    """
    Used by `DataClient`, to send shares of its data to the other party
    """

    MUL_OwnVal_SHARE = indexer.auto()
    """
    Used by `DataClient`, send its A - U to the other party 
    (A is its share of its own matrix, and U is its share of its triple)
    """

    MUL_OtherVal_SHARE = indexer.auto()
    """
    Used by `DataClient`, send its B - V to the other party 
    (B is its share of the other party's matrix, and V is its share of the other party's triple)
    """
    MUL_OUT_SHARE = indexer.auto()

    MUL_Mat_Share = indexer.auto()
    MUL_AsubU_Share = indexer.auto()
    MUL_BsubV_Share =indexer.auto()

    PRED_LABEL = indexer.auto()
    PRED_GRAD = indexer.auto()
    CLIENT_OUT_GRAD = indexer.auto()
    CLIENT_PARA_UPDATE = indexer.auto()
    CLIENT_ROUND_OVER = indexer.auto()

    ALIGN_AES_KEY = indexer.auto()
    ALIGN_ENC_IDS = indexer.auto()
    ALIGN_FINAL_IDS = indexer.auto()

    # Those is for message responses
    RECEIVED_ERR = indexer.auto()
    RECEIVED_OK = indexer.auto()


class ReceiveERRType(Enum):
    BUFFER_OCCUPIED = 1
    UNRECOGNIZED_SENDER = 2
    CONNECTION_FAILED = 3


class PackedMessage:
    """
    Message class
    """
    def __init__(self, header: MessageType, data, key=None):
        """
        :param header:  Header of the message, specify the type of message
        :param data: Message data

        Initialize a Message
        """
        self.header = header
        self.data = data
        self.key = key

    def __str__(self):
        return "header:" + self.header.__str__() + "\ndata:" + self.data.__str__() + "\n" + self.key.__str__() + "\n"
