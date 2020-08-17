import numpy as np
import sys
import pickle
import grpc
import time
import datetime
import concurrent.futures as futures
from Communication.Channel import BaseChannel
from Communication.protobuf import message_pb2, message_pb2_grpc
from Communication.Message import MessageType, PackedMessage
from Utils.Log import Logger


def encode_ComputationData(computation_message: PackedMessage):
    return message_pb2.ComputationData(
        type=computation_message.header.value,
        python_bytes=pickle.dumps((computation_message.data, computation_message.key)))

def decode_ComputationData(computation_data: message_pb2.ComputationData):
    data, key = pickle.loads(computation_data.python_bytes)
    return PackedMessage(MessageType(computation_data.type), data, key)


class ComputationRPCClient:
    def __init__(self, address):
        self.channel = grpc.insecure_channel(address, options=[('grpc.max_receive_message_length', 10 * 1024 * 1024)])
        self.stub = message_pb2_grpc.MPCServiceStub(self.channel)

    def sendComputationMessage(self, computation_message, client_id: int, time_out: float):
        msg = encode_ComputationData(computation_message)
        msg.client_id = client_id
        response = self.stub.GetComputationData(msg, time_out)
        return decode_ComputationData(response)


class ComputationServicer(message_pb2_grpc.MPCServiceServicer):
    def __init__(self, msg_handler):
        self.msg_handler = msg_handler

    def GetComputationData(self, request, context):
        msg = decode_ComputationData(request)
        if self.msg_handler(msg, request.client_id):
            return encode_ComputationData(PackedMessage(header=MessageType.RECEIVED_OK, data=None))
        else:
            return encode_ComputationData(PackedMessage(header=MessageType.RECEIVED_ERR, data="Buffer occupied, send it later"))


class ComputationRPCServer:
    def __init__(self, port, max_workers, msg_handler):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers),
                                  options=[('grpc.max_receive_message_length', 50 * 1024 * 1024)])
        message_pb2_grpc.add_MPCServiceServicer_to_server(ComputationServicer(msg_handler), self.server)
        self.server.add_insecure_port(port)

    def start(self):
        self.server.start()


class Peer(BaseChannel):
    """
    客户端类。每个客户端也包含一个服务器用于监听其他客户端。
    """
    def __init__(self, self_id, self_port: str, self_max_workers: int, ip_dict: dict, time_out: float=1,
                 logger: Logger=None):
        """
        :param self_id: 客户端的ID，必须是唯一的
        :param self_port: 客户端监听的端口号
        :param self_max_workers:
        :param ip_dict: 一个dict，指定客户端id和ip地址的对应关系。
        :param time_out:
        """
        super(Peer, self).__init__(self_id, "", len(ip_dict), logger)
        self.server = ComputationRPCServer(self_port, self_max_workers, self.buffer_msg)
        self.rpc_clients = [ComputationRPCClient(ip_dict[client_num]) for client_num in ip_dict]
        self.ip_dict = ip_dict
        self.receive_buffer = [dict() for _ in ip_dict]
        self.time_out = time_out
        self.server.start()
        time.sleep(1)
        self.logger.log("Peer id {} started. on port {}".format(self.client_id, self_port))

    def buffer_msg(self, msg: PackedMessage, sender_id):
        """
        将接收到的信息进行缓存
        :param msg:
        :param sender_id:
        :return: True: 表示消息已经被接收；False：表示当前缓存中有消息未被取出；None: 表示未记录的发送者。
        """
        if sender_id in self.ip_dict:
            if self.receive_buffer[sender_id].get(msg.key) is None:
                self.receive_buffer[sender_id][msg.key] = []
            self.receive_buffer[sender_id][msg.key].append(msg)
            return True
        self.logger.logE("Received from id {} not in the ip_dict {}".format(sender_id, self.ip_dict))
        return False

    def receive(self, sender: int, time_out: float, key=None, **kwargs) -> PackedMessage:
        """
        :param sender: 发送方
        :param time_out: 接收的延时。如果未设置则采用默认延时。
        :return: None: 表示未收到消息。否则是收到的消息。
        """
        if not time_out:
            time_out = self.time_out
        start_receive_time = time.time()
        if self.receive_buffer[sender].get(key) is None:
            self.receive_buffer[sender][key] = []
        key_buffer = self.receive_buffer[sender][key]
        try_interval = kwargs.get("interval") or 0.001

        while len(key_buffer) == 0:
            time.sleep(try_interval)
            key_buffer = self.receive_buffer[sender].get(key)
            if time.time() - start_receive_time > time_out:
                self.logger.log("Timeout while receiving from client %d, key %s. Time elapsed %.3f" % (sender, str(key), time_out))
                return None
        msg = key_buffer[0]
        del key_buffer[0]
        return msg

    def send(self, receiver: int, msg: PackedMessage, time_out: float=None):
        """
        :param receiver: 接收方
        :param msg: 消息
        :param time_out: 发送的最大时长，超出该时长未发送成功则停止。如果未设置则采用默认延时。
        :return:
        """
        if not time_out:
            time_out = self.time_out
        resp = PackedMessage(MessageType.RECEIVED_ERR, None)
        start_send_time = time.time()
        while resp.header == MessageType.RECEIVED_ERR:
            if time.time() - start_send_time > time_out:
                self.logger.log("Timeout while sending to client %d. Time elapsed %.3f" % (receiver, time_out))
                return resp
            try:
                resp = self.rpc_clients[receiver].sendComputationMessage(msg, self.client_id, time_out)
            except Exception as e:
                self.logger.logE("Error while sending to client %d" % receiver + ":" + str(e))
                return resp
        return resp

    def reset(self):
        """
        Clear all messages in the buffer
        :return:
        """
        for buffer in self.receive_buffer:
            buffer.clear()