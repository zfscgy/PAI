import grpc
import pickle
import enum
from Communication.protobuf import message_pb2_grpc, message_pb2
from Communication.protobuf.message_pb2_grpc import QueryMPCTaskStub
from Communication.protobuf.message_pb2 import TaskQuery


class TaskQueryStatus(enum.Enum):
    ok = 0
    err = 1


class QueryMPCTaskServicer(message_pb2_grpc.QueryMPCTaskServicer):
    def __init__(self, task):
        self.task = task
        self.query_dict = dict()

    def QueryTask(self, request: message_pb2.TaskQuery, context):
        def encode(status: TaskQueryStatus, obj=None):
            return message_pb2.TaskResponse(status=status.value, python_bytes=pickle.dumps(obj))
        if request.query_string not in self.query_dict:
            return encode(TaskQueryStatus.err, "Query url not exist")
        try:
            resp = self.query_dict[request.query_string]()
            return encode(TaskQueryStatus.ok, resp)
        except Exception as e:
            return encode(TaskQueryStatus.err, str(e))

    def add_query(self, url, func):
        if callable(func):
            self.query_dict[url] = func
            return True
        else:
            return False

    def add_query_dict(self, query_dict):
        for key in query_dict:
            if not callable(query_dict[key]):
                return False
        self.query_dict.update(query_dict)
        return True


def add_query_service_to_computation_grpc_server(task):
    from Communication.RPCComm import Peer
    peer = task.channel
    if not isinstance(peer, Peer):
        return None
    task_servicer = QueryMPCTaskServicer(task)
    message_pb2_grpc.add_QueryMPCTaskServicer_to_server(task_servicer, peer.server.server)
    return task_servicer


def decode_query(msg: TaskQuery):
    return TaskQueryStatus(msg.status).name, pickle.loads(msg.python_bytes)


class TaskQueryClient:
    def __init__(self, addr):
        channel = grpc.insecure_channel(addr)
        self.stub = QueryMPCTaskStub(channel)

    def query(self, url):
        resp = self.stub.QueryTask(url)
        return {
            "status": TaskQueryStatus(resp.status).name,
            "msg": pickle.loads(resp.python_bytes)
        }
