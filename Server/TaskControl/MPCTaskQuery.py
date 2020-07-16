import grpc
import pickle
from Communication.protobuf.message_pb2_grpc import QueryMPCTaskStub
from Communication.protobuf.message_pb2 import TaskQuery
from Server.TaskControl.MPCTask import TaskQueryStatus


def decode_query(msg: TaskQuery):
    return TaskQueryStatus(msg.status).name, pickle.loads(msg.python_bytes)


class MPCTaskQueryClient:
    def __init__(self, addr):
        channel = grpc.insecure_channel(addr)
        self.stub = QueryMPCTaskStub(channel)

    def query(self, url):
        resp = self.stub.QueryTask(url)
        return {
            "status": TaskQueryStatus(resp.status).name,
            "msg": pickle.loads(resp.python_bytes)
        }
