import threading
from flask import Flask, request
import requests
import json
import os

from Communication.protobuf.message_pb2 import TaskQuery
from Utils.Log import Logger
from Server.TaskControl.MPCTaskQuery import MPCTaskQueryClient

main_server = Flask(__name__)

from Server.HttpServer.ServerConfig import ServerLogPath, ServerTaskRoot, ClientProtocol
from Server.HttpServer.TaskParaGenerator import TaskParaGenerator, generate_paras


logger = Logger(open(ServerLogPath + "/main_server-%d_log.txt" % id(main_server), "a"), level=0)


def resp_msg(status="ok", msg=None):
    return {
        "status": status,
        "msg": msg
    }


@main_server.route("/helloWorld")
def hello_world():
    return "Hello, world"


@main_server.route("/createTask", methods=["POST"])
def create_task():
    try:
        post_data = json.loads(request.get_data(as_text=True))
        logger.log("Received createTask request, with data:\n" + json.dumps(post_data, indent=2))
    except:
        logger.logE("Error while parsing post json data. Stop.")
        return resp_msg("err", "Error while parsing post json data")

    if "task_name" not in post_data:
        err = "post_data must have key task_name"
        logger.logE(err)
        return resp_msg("err", err)

    # Save json file
    task_name = post_data["task_name"]
    if os.path.isfile(ServerTaskRoot + task_name + ".json"):
        logger.logE("Task with name {} already exists".format(task_name))
        return resp_msg("err", "Task with name {} already exists".format(task_name))

    # Generate client task parameters
    task_paras = generate_paras(post_data)
    if isinstance(task_paras, str):
        logger.logE("Generate task parameters from post_json failed: " + task_paras)
        return resp_msg("err", "Generate task parameters from post_json failed: " + task_paras)

    client_task_paras = task_paras.client_paras
    client_http_addrs = task_paras.http_dict

    post_errors = {}
    def post_to_client_webserver(client_id):
        client_task_para = client_task_paras[client_id]
        try:
            resp = requests.post(ClientProtocol + client_http_addrs[client_id] + "/createTask", json=client_task_para)
        except Exception as e:
            err = "Post to client /createTask failed, error {}".format(e)
            logger.logE(err)
            post_errors[client_id] = err
            return
        if resp.status_code != requests.codes.ok:
            err = "Post to client /createTask failed, with status code %d" % resp.status_code
            logger.logE(err)
            post_errors[client_id] = True
        else:
            resp_msg = resp.json()
            if resp_msg["status"] != "ok":
                err = "Received error response from client {}, message {}".format(client_id, resp_msg)
                logger.logE(err)
                post_errors[client_id] = err

    posting_threads = []
    for client_id in range(len(client_task_paras)):
        posting_threads.append(threading.Thread(target=post_to_client_webserver, args=(client_id,),
                                                name="Thread-PostCreateTask-to-Client-%d" % client_id))
        posting_threads[-1].start()

    for posting_thread in posting_threads:
        posting_thread.join()

    if len(post_errors) != 0:
        err = "Failed to post to client /createTask. Error state of clients:" + str(post_errors)
        logger.logE(err)
        return resp_msg("err", err)

    json.dump(post_data, open(ServerTaskRoot + task_name + ".json", "w+"), indent=4)
    logger.log("Task Created, Save json file in " + ServerTaskRoot + task_name + ".json")
    return resp_msg()


task_query_dict = dict()


@main_server.route("/startTask", methods=["GET"])
def start_task():
    task_name = request.args.get("task_name")
    try:
        task_json = json.load(open(ServerTaskRoot + task_name + ".json", "r"))
    except Exception as e:
        err = "Load task error: {}".format(e)
        logger.logE("Cannot load task:\n" + err)
        return resp_msg("err", err)

    task_para_generator = TaskParaGenerator(task_json)
    task_para_generator.generate_paras()
    client_addrs = task_para_generator.http_dict
    client_grpc_addrs = task_para_generator.ip_dict

    startTask_errors = {}
    def request_to_clients(client_id):
        try:
            resp = requests.get(ClientProtocol + client_addrs[client_id] + "/startTask",
                                {"task_name": task_name, "client_id": client_id})
        except Exception as e:
            err = "Get request for client startTask failed with client id %d, Error:\n".format(client_id) + str(e)
            logger.logE(err)
            startTask_errors[client_id] = err
            return
        if resp.status_code != requests.codes.ok:
            err = "Get request for client startTask failed with status code {}, client id {}".\
                       format(resp.status_code, client_id)
            logger.logE(err)
            startTask_errors[client_id] = err
            return
        resp = resp.json()
        if resp["status"] != "ok":
            err = "Get request for client startTask failed with error message {}, client id {}".format(resp["msg"], client_id)
            logger.logE(err)
            startTask_errors[client_id] = err
            return

    request_tasks = []
    for i, addr in enumerate(client_addrs):
        request_tasks.append(threading.Thread(target=request_to_clients, args=(i,)))
        request_tasks[-1].start()
    for request_task in request_tasks:
        request_task.join()

    if len(startTask_errors) == 0:
        grpc_clients = [
            MPCTaskQueryClient(client_grpc_addrs[i]) for i in range(len(client_grpc_addrs))
        ]
        task_query_dict[task_name] = grpc_clients
        return resp_msg()

    else:
        error_msgs = "Get request for client startTask failed with message:\n" + str(startTask_errors)
        logger.log(error_msgs)
        return resp_msg("err", error_msgs)


@main_server.route("/queryTask", methods=["GET"])
def query_task():
    task_name = request.args.get("task_name")
    query = request.args.get("query")
    client_id = request.args.get("client", type=int)
    if task_name is None or query is None:
        return resp_msg("err", "Url arguments missing, must have task_name and query")
    if task_name not in task_query_dict:
        return resp_msg("err", "Task not exist")
    client_id = client_id or 0
    if client_id >= len(task_query_dict[task_name]):
        return resp_msg("err", "Client not exist")
    try:
        res = task_query_dict[task_name][client_id].query(TaskQuery(query_string=query))
        logger.log("Answered query. task name: {} url: {}, client: {}, res: {}".format(task_name, query, client_id, res))
        return resp_msg("ok", res)
    except Exception as e:
        logger.logE("GRPC query failed: " + str(e))
        return resp_msg("err", "Query client grpc failed")
