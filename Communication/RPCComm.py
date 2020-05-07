import numpy as np
import pickle
import grpc
import time
import concurrent.futures as futures
from Communication.protobuf import message_pb2, message_pb2_grpc
from Communication.Message import MessageType, ComputationMessage


class BaseChannel:
    """
    Channel for communication
    """
    def __init__(self, n_clients: int):
        """
        :param n_clients: Number of clients that will Join this channel
        """
        self.n_clients = n_clients

    def send(self, receiver: int, msg: ComputationMessage):
        """
        :return: `True` if message is send successfully. If the port is occupied, wait for time_outs.
        """
        raise NotImplementedError()

    def receive(self, sender: int):
        raise NotImplementedError()


def encode_ComputationData(computation_message: ComputationMessage):
    return message_pb2.ComputationData(type=computation_message.header.value,
                                       python_bytes=pickle.dumps(computation_message.data))


def decode_ComputationData(computation_data: message_pb2.ComputationData):
    return ComputationMessage(MessageType(computation_data.type), pickle.loads(computation_data.python_bytes))


class ComputationRPCClient:
    def __init__(self, address):
        self.channel = grpc.insecure_channel(address)
        self.stub = message_pb2_grpc.MPCServiceStub(self.channel)

    def sendComputationMessage(self, computation_message, client_id):
        msg = encode_ComputationData(computation_message)
        msg.client_id = client_id
        response = self.stub.GetComputationData(msg)
        return decode_ComputationData(response)


class ComputationServicer(message_pb2_grpc.MPCServiceServicer):
    def __init__(self, msg_handler):
        self.msg_handler = msg_handler

    def GetComputationData(self, request, context):
        msg = decode_ComputationData(request)
        self.msg_handler(msg, request.client_id)
        return encode_ComputationData(ComputationMessage(header=MessageType.RECEIVED_OK, data=None))


class ComputationRPCServer:
    def __init__(self, port, max_workers, msg_handler):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        message_pb2_grpc.add_MPCServiceServicer_to_server(ComputationServicer(msg_handler), self.server)
        self.server.add_insecure_port(port)

    def start(self):
        self.server.start()


class Peer(BaseChannel):
    def __init__(self, self_id, self_port: str, self_max_workers: int, ip_dict: dict, time_out: float=1):
        super(Peer, self).__init__(len(ip_dict))
        self.server = ComputationRPCServer(self_port, self_max_workers, self.handle_msg)
        self.rpc_clients = [ComputationRPCClient(ip_dict[client_num]) for client_num in ip_dict]
        self.client_id = self_id
        self.ip_dict = ip_dict
        self.receive_buffer = [None for _ in ip_dict]
        self.time_out = time_out
        self.server.start()

    def handle_msg(self, msg, sender_id):
        if sender_id in self.ip_dict:
            msg_received_time = time.time()
            while self.receive_buffer[sender_id] is not None:
                time.sleep(0.01)
                if time.time() > msg_received_time + self.time_out:
                    return False

            self.receive_buffer[sender_id] = msg
            return True
        return None

    def receive(self, sender: int):
        start_receive_time = time.time()
        while self.receive_buffer[sender] is None:
            time.sleep(0.01)
            if time.time() - start_receive_time > self.time_out:
                return None
        msg = self.receive_buffer[sender]
        self.receive_buffer[sender] = None
        return msg

    def send(self, receiver: int, msg: ComputationMessage):
        resp = self.rpc_clients[receiver].sendComputationMessage(msg, self.client_id)
        return resp