import numpy as np
from enum import Enum


def log(msg):
    print(msg)


class MessageType(Enum):
    NULL = 99
    """
    
    """
    DATA_DIM = 1

    """
    该指令表示发送一个设置矩阵三元组的指令给三元组生成节点
    其data为(target_client, shape_sender, shape_target)
    分别表示希望把三元组的另一个share发送给哪个节点，自己的矩阵的shape，对方的矩阵的shape
    该三元组主要用于矩阵相乘
    """
    SET_TRIPLET = 11

    """
    
    """
    TRIPLE_ARRAY = 12
    """
    该指令表示分享batch_data数组给特定的client，其data为np.ndarray
    """
    MUL_DATA_SHARE = 20
    """
    """
    MUL_OwnVal_SHARE = 21
    """
    """
    MUL_OtherVal_SHARE = 22


class Message:
    def __init__(self, header, data):
        self.header = header
        self.data = data

class Channel:
    def __init__(self, n_clients: int):
        self.n_clients = n_clients
        self.port = [None for _ in range(n_clients * n_clients)]
        self.triplets_id = None

    def send(self, sender: int, receiver: int, msg: Message):
        port_num = sender * self.n_clients + receiver
        if self.port[port_num] is not None:
            log("Receiver %d's port is occupied" % receiver)
            return False
        self.port[port_num] = msg
        return True

    def receive(self, receiver, sender):
        port_num = sender * self.n_clients + receiver
        msg = self.port[port_num]
        self.port[port_num] = None
        return msg


class BaseClient:
    def __init__(self, client_id, channel):
        self.client_id = client_id
        self.channel = channel

    def recv_msg(self, sender):
        return self.channel.receive(self.client_id, sender)

    def send_msg(self, receiver, msg: Message):
        self.channel.send(self.client_id, receiver, msg)

    def receive_msg(self, sender):
        pass


class TripletsProvider(BaseClient):
    def __init__(self, client_id: int, channel: Channel):
        super(TripletsProvider, self).__init__(client_id, channel)
        self.channel.triplets_id = self.client_id
        self.triplet_proposals = dict()

    def receive_msg(self, sender):
        msg = self.channel.receive(self.client_id, sender)
        if msg.header == MessageType.SET_TRIPLET:
            target = msg.data[0]
            shape_sender = msg.data[1]
            shape_target = msg.data[2]
            existing_shapes = self.triplet_proposals.get((target, sender))
            if existing_shapes is not None:
                if existing_shapes[0] == shape_target and existing_shapes[1] == shape_sender:
                    self.generate_and_send_triplets((sender, target), (shape_sender, shape_target))
                    del self.triplet_proposals[(target, sender)]
                else:
                    log("Triplet shapes not match with target client")
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
        self.shared_out = [None for _ in range(channel.n_clients)]

    def receive_msg(self, sender):
        msg = self.channel.receive(self.client_id, sender)
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

    def get_next_batch(self):
        self.batch_data = np.random.uniform(-1, 1, [self.batch_size, self.data_dim])

    def send_data_sim(self, receiver):
        self.send_msg(receiver, Message(MessageType.DATA_DIM, self.data_dim))

    # 提供数据作为矩阵乘法中的乘数
    def set_triplet_AB(self, other_id):
        self.send_msg(self.channel.triplets_id,
                      Message(MessageType.SET_TRIPLET, (other_id, self.batch_data.shape,
                                                        self.para.shape)))

    # 提供参数作为矩阵乘法中的的被乘数
    def set_triplet_BA(self, other_id):
        self.send_msg(self.channel.triplets_id,
                      Message(MessageType.SET_TRIPLET, (other_id, self.other_paras[other_id].shape,
                                                        (self.batch_size, self.other_paras[other_id].shape[0]))))

    def share_data(self, other_id):
        self.shared_own_mat[other_id] = self.batch_data * np.random.uniform(0, 1, self.batch_data.shape)
        self.send_msg(other_id, Message(MessageType.MUL_DATA_SHARE, self.batch_data - self.shared_own_mat[other_id]))

    def share_para(self, other_id):
        self.shared_own_mat[other_id] = self.other_paras[other_id] * \
                                        np.random.uniform(0, 1, self.other_paras[other_id].shape)
        self.send_msg(other_id, Message(MessageType.MUL_DATA_SHARE, self.other_paras[other_id] - self.shared_own_mat[other_id]))

    def recover_own_value(self, other_id):
        self.send_msg(other_id, Message(MessageType.MUL_OwnVal_SHARE,
                                        self.shared_own_mat[other_id] - self.current_triplets[other_id][0]))

    def recover_other_value(self, other_id):
        self.send_msg(other_id, Message(MessageType.MUL_OtherVal_SHARE,
                                        self.shared_other_mat[other_id] - self.current_triplets[other_id][1]))

    def get_shared_out_AB(self, other_id):
        self.shared_out[other_id] = - np.matmul(self.recovered_own_value[other_id],
                                                self.recovered_other_value[other_id])
        self.shared_out[other_id] += np.matmul(self.shared_own_mat[other_id], self.recovered_other_value[other_id]) +\
            np.matmul(self.recovered_own_value[other_id], self.shared_other_mat[other_id]) + self.current_triplets[other_id][2]

    def get_shared_out_BA(self, other_id):
        self.shared_out[other_id] = np.matmul(self.recovered_other_value[other_id], self.shared_own_mat[other_id]) +\
            np.matmul(self.shared_other_mat[other_id], self.recovered_own_value[other_id]) + self.current_triplets[other_id][2]