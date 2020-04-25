import numpy as np
from enum import Enum
import threading
import time

def log(msg):
    print(msg + "\n")


class MessageType(Enum):
    NULL = 99

    # 以下指令主要用于Secret-sharing矩阵乘法，主要场景为一个客户端提供数据，另一个提供参数，求其变换后的数据
    # 比如：一个客户端的数据shape = (batch_size, dim)，另一个客户端拥有对应的参数 (dim, out_dim)
    # 该方法可以计算出这两个矩阵的乘积，而客户端的自己的数据不需要发送出去。

    DATA_DIM = 1
    """
    Dimension message, the `data` should be:
    dim:int
    """

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



class Message:
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


class Channel:
    """
    Channel for communication
    """
    def __init__(self, n_clients: int):
        """
        :param n_clients: Number of clients that will Join this channel
        """
        self.n_clients = n_clients
        self.port = [None for _ in range(n_clients * n_clients)]
        self.triplets_id = None
        self.time_out = 1000

    def send(self, sender: int, receiver: int, msg: Message):
        """
        :return: `True` if message is send successfully. If the port is occupied, wait for time_outs.
        """
        port_num = sender * self.n_clients + receiver
        time_start = time.time()
        while self.port[port_num] is not None:
            if time.time() - time_start > self.time_out / 1000:
                break
        if self.port[port_num] is not None:
            return False
        else:
            self.port[port_num] = msg
            return True

    def receive(self, receiver: int, sender: int):
        """
        :return: The message from sender. If there's no message during the waiting time, return `None`
        """
        port_num = sender * self.n_clients + receiver
        msg = None
        time_start = time.time()
        while msg is None:
            msg = self.port[port_num]
            if time.time() - time_start > self.time_out / 1000:
                break
        self.port[port_num] = None
        return msg


class BaseClient:
    """
    Base class of client
    """
    def __init__(self, client_id, channel):
        """
        :param client_id: An integer to identify the client
        :type client_id: int
        :param channel: Channel for communication
        """
        self.client_id = client_id
        self.channel = channel

    def send_msg(self, receiver: int, msg: Message):
        """
        Send a message to the receiver

        :return: `True` or `False`
        """
        self.channel.send(self.client_id, receiver, msg)

    def receive_msg(self, sender):
        pass


class TripletsProvider(BaseClient):
    """
    A client for generate beaver triples. It can listen to other clients
    """
    def __init__(self, client_id: int, channel: Channel):
        super(TripletsProvider, self).__init__(client_id, channel)
        self.channel.triplets_id = self.client_id
        self.triplet_proposals = dict()
        self.listening_thread = [None for _ in range(channel.n_clients)]
        self.listening = False

    def receive_msg(self, sender):
        log("Triplet Provider: Started Listening on client %d" % sender)
        while True and self.listening:
            msg = self.channel.receive(self.client_id, sender)
            if msg is None:
                continue
            if msg.header == MessageType.SET_TRIPLET:
                target = msg.data[0]
                shape_sender = msg.data[1]
                shape_target = msg.data[2]
                existing_shapes = self.triplet_proposals.get((target, sender))
                if existing_shapes is not None:
                    if existing_shapes[0] == shape_target and existing_shapes[1] == shape_sender:
                        self.generate_and_send_triplets((sender, target), (shape_sender, shape_target))
                        log("Triplet provider: send triples")
                        del self.triplet_proposals[(target, sender)]
                    else:
                        log("Triplet provider: Triplet shapes not match with target client")
                else:
                    self.triplet_proposals[(sender, target)] = (shape_sender, shape_target)

    def generate_and_send_triplets(self, clients, shapes):
        # 判断哪一个是乘数，哪一个是被乘数
        if shapes[1][0] != shapes[0][1]:
            shapes = [shapes[1], shapes[0]]
            clients = [clients[1], clients[0]]
        u0 = np.random.uniform(-1, 1, shapes[0])
        u1 = np.random.uniform(-1, 1, shapes[0])
        v0 = np.random.uniform(-1, 1, shapes[1])
        v1 = np.random.uniform(-1, 1, shapes[1])
        z = np.matmul(u0 + u1, v0 + v1)
        z0 = z * np.random.uniform(0, 1, z.shape)
        z1 = z - z0
        self.send_msg(clients[0], Message(MessageType.TRIPLE_ARRAY, (clients[1], u0, v0, z0)))
        self.send_msg(clients[1], Message(MessageType.TRIPLE_ARRAY, (clients[0], v1, u1, z1)))

    def start_listening(self):
        """
        Start the listening thread
        """
        self.listening = True
        for i in range(self.channel.n_clients):
            self.listening_thread[i] = threading.Thread(target=self.receive_msg, args=(i,))
            self.listening_thread[i].start()

    def stop_listening(self):
        """
        Stop the listening thread
        """
        self.listening = False
        for i in range(self.channel.n_clients):
            self.listening_thread[i].join()

class DataClient(BaseClient):
    def __init__(self, client_id: int, channel: Channel, batch_size: int, data_dim: int, output_dim: int):
        # random generate some to data
        super(DataClient, self).__init__(client_id, channel)
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.data_dim = data_dim
        self.para = np.random.uniform(-1, 1, [data_dim, output_dim])
        self.batch_data = None

        # 变量储存器，用于Secret Sharing矩阵乘法
        self.current_triplets = [None for _ in range(channel.n_clients)]
        self.other_paras = [None for _ in range(channel.n_clients)]

        self.shared_own_mat = [None for _ in range(channel.n_clients)]
        self.shared_other_mat = [None for _ in range(channel.n_clients)]

        self.recovered_own_value = [None for _ in range(channel.n_clients)]
        self.recovered_other_value = [None for _ in range(channel.n_clients)]
        self.shared_out_AB = [None for _ in range(channel.n_clients)]
        self.shared_out_BA = [None for _ in range(channel.n_clients)]

        self.calc_threads = [None for _ in range(channel.n_clients)]
        self.working = True

    def receive_msg(self, sender):
        """
        Receive message from sender.

        """
        msg = None
        while msg is None and self.working:
            msg = self.channel.receive(self.client_id, sender)
        if msg is None:
            log("Client id %d :Work Terminated")
            return
        if msg.header == MessageType.DATA_DIM:
            self.other_paras[sender] = np.random.uniform(-1, 1, [msg.data, self.output_dim])
        elif msg.header == MessageType.MUL_DATA_SHARE:
            self.shared_other_mat[sender] = msg.data
        elif msg.header == MessageType.TRIPLE_ARRAY:
            self.current_triplets[msg.data[0]] = msg.data[1:]
        elif msg.header == MessageType.MUL_OwnVal_SHARE:
            self.recovered_other_value[sender] = self.shared_other_mat[sender] - self.current_triplets[sender][1] + msg.data
        elif msg.header == MessageType.MUL_OtherVal_SHARE:
            self.recovered_own_value[sender] = self.shared_own_mat[sender] - self.current_triplets[sender][0] + msg.data

    def __calculate_first_hidden_layer(self, other_id):
        """
        :param other_id:
        :return:
        """
        def get_next_batch():
            self.batch_data = np.random.uniform(-1, 1, [self.batch_size, self.data_dim])

        def send_data_dim():
            self.send_msg(other_id, Message(MessageType.DATA_DIM, self.data_dim))

        # 提供数据作为矩阵乘法中的乘数
        def set_triplet_AB():
            self.send_msg(self.channel.triplets_id,
                          Message(MessageType.SET_TRIPLET, (other_id, self.batch_data.shape,
                                                            self.para.shape)))

        # 提供参数作为矩阵乘法中的的被乘数
        def set_triplet_BA():
            self.send_msg(self.channel.triplets_id,
                          Message(MessageType.SET_TRIPLET, (other_id, self.other_paras[other_id].shape,
                                                            (self.batch_size, self.other_paras[other_id].shape[0]))))

        def share_data():
            self.shared_own_mat[other_id] = self.batch_data * np.random.uniform(0, 1, self.batch_data.shape)
            self.send_msg(other_id, Message(MessageType.MUL_DATA_SHARE, self.batch_data - self.shared_own_mat[other_id]))

        def share_para():
            self.shared_own_mat[other_id] = self.other_paras[other_id] * \
                                            np.random.uniform(0, 1, self.other_paras[other_id].shape)
            self.send_msg(other_id, Message(MessageType.MUL_DATA_SHARE, self.other_paras[other_id] - self.shared_own_mat[other_id]))

        def recover_own_value():
            self.send_msg(other_id, Message(MessageType.MUL_OwnVal_SHARE,
                                            self.shared_own_mat[other_id] - self.current_triplets[other_id][0]))

        def recover_other_value():
            self.send_msg(other_id, Message(MessageType.MUL_OtherVal_SHARE,
                                            self.shared_other_mat[other_id] - self.current_triplets[other_id][1]))

        def get_shared_out_AB():
            self.shared_out_AB[other_id] = - np.matmul(self.recovered_own_value[other_id],
                                                       self.recovered_other_value[other_id])
            self.shared_out_AB[other_id] += np.matmul(self.shared_own_mat[other_id], self.recovered_other_value[other_id]) + \
                                            np.matmul(self.recovered_own_value[other_id], self.shared_other_mat[other_id]) + self.current_triplets[other_id][2]

        def get_shared_out_BA():
            self.shared_out_BA[other_id] = np.matmul(self.recovered_other_value[other_id], self.shared_own_mat[other_id]) + \
                                           np.matmul(self.shared_other_mat[other_id], self.recovered_own_value[other_id]) + self.current_triplets[other_id][2]

        # Calculate X_own * Theta_other
        def calc_AB():
            log("Client id:%d, start calc_AB" % self.client_id)
            set_triplet_AB()
            self.receive_msg(self.channel.triplets_id)
            log("Client id:%d, triple data get" % self.client_id)
            share_data()
            self.receive_msg(other_id)
            log("Client id:%d, para share get" % self.client_id)
            recover_own_value()
            self.receive_msg(other_id)
            log("Client id:%d, own value recovered" % self.client_id)
            recover_other_value()
            self.receive_msg(other_id)
            log("Client id:%d, own value recovered" % self.client_id)
            get_shared_out_AB()
            log("Client id:%d, finished calc_AB" % self.client_id)

        # Calculate Theta_own * X_other
        def calc_BA():
            log("Client id:%d, start calc_BA" % self.client_id)
            set_triplet_BA()
            self.receive_msg(self.channel.triplets_id)
            log("Client id:%d, triple data get" % self.client_id)
            share_para()
            self.receive_msg(other_id)
            log("Client id:%d, data share get" % self.client_id)
            recover_own_value()
            self.receive_msg(other_id)
            log("Client id:%d, own value recovered" % self.client_id)
            recover_other_value()
            self.receive_msg(other_id)
            log("Client id:%d, other value recovered" % self.client_id)
            get_shared_out_BA()
            log("Client id:%d, finished calc_BA" % self.client_id)

        get_next_batch()
        send_data_dim()
        self.receive_msg(other_id)

        if other_id < self.client_id:
            calc_AB()
            calc_BA()
        else:
            calc_BA()
            calc_AB()

    def start_calc_first_layer(self, other_id):
        self.working = True
        thread = threading.Thread(target=self.__calculate_first_hidden_layer, args=(other_id,))
        self.calc_threads[other_id] = thread
        thread.start()
        return thread

    def stop_work(self):
        self.working = False