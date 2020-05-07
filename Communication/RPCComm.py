import numpy as np
import pickle
import grpc
import time
import concurrent.futures as futures
from Communication.protobuf import message_pb2, message_pb2_grpc
from Client import MessageType, ComputationMessage, BaseClient


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
    return ComputationMessage(MessageType(computation_data.type), pickle.loads(computation_data.numpy_bytes))


class ComputationRPCClient:
    def __init__(self, address):
        self.channel = grpc.insecure_channel(address)
        self.stub = message_pb2_grpc.MPCServiceStub(self.channel)

    def sendComputationMessage(self, msgType, python_data):
        msg = message_pb2.ComputationData(type=msgType, python_bytes=pickle.dumps(python_data))
        response = self.stub.GetComputationData(msg)
        return response


class ComputationServicer(message_pb2_grpc.MPCServiceServicer):
    def __init__(self, msg_handler):
        self.msg_handler = msg_handler

    def GetComputationData(self, request, context):
        msg = decode_ComputationData(request)
        self.msg_handler(msg, context.peer())
        return encode_ComputationData(ComputationMessage(header=MessageType.RECEIVED_OK, data=None))


class ComputationRPCServer:
    def __init__(self, port, max_workers, msg_handler):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        message_pb2_grpc.add_MPCServiceServicer_to_server(ComputationServicer(msg_handler), self.server)
        self.server.add_secure_port(port)

    def start(self):
        self.server.start()


class Peer(BaseChannel):
    def __init__(self, server_addresses: str, self_port: int, self_max_workers: int, ip_dict: dict, time_out: float=1):
        super(Peer, self).__init__(len(ip_dict))
        self.server = ComputationRPCServer(self_port, self_max_workers, self.handle_msg)
        self.clients = [ComputationRPCClient(address) for address in server_addresses]
        self.ip_dict = ip_dict
        self.receive_buffer = [None for _ in len(ip_dict)]
        self.time_out = time_out

    def handle_msg(self, msg, address):
        if address in self.ip_dict:
            sender = self.ip_dict[address]
            msg_received_time = time.time()
            while self.receive_buffer[sender] is None:
                time.sleep(0.01)
                if time.time() > msg_received_time + self.time_out:
                    return False

            self.receive_buffer[sender] = msg
            return True
        return None

    def send(self, receiver: int, msg: ComputationMessage):


