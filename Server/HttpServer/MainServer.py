import threading
from flask import Flask, request
import requests
import json
import os
from Communication.protobuf.message_pb2 import TaskQuery
from Utils.Log import Logger
from Task.TaskQuery import TaskQueryClient
from Server.HttpServer.ServerConfig import ServerLogPath, ServerTaskRoot, ClientProtocol
from Server.HttpServer.TaskParaGenerator import generate_task_paras, generate_dataset_json
from Server.HttpServer.BroadcastRequests import broadcast_request


main_server = Flask(__name__)

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
    try:
        client_paras, client_http_addrs = generate_task_paras(post_data)
        data_json = generate_dataset_json(post_data)
    except Exception as e:
        err = "Generate task parameters failed: " + str(e)
        logger.logE(err)
        return resp_msg("err", err)

    global http_query_dict
    http_query_dict[task_name] = client_http_addrs

    client_errs, client_resps = broadcast_request([ClientProtocol + client_addr + "/createTask" for client_addr in client_http_addrs],
                                                  "post", jsons=client_paras)
    if len(client_errs) != 0:
        err = "CreateTask failed due to POST error: " + str(client_errs)
        logger.logE(err)
        return resp_msg("err", err)

    err_status = False
    for i, client_resp in enumerate(client_resps):
        if client_resp["status"] != "ok":
            err = "Create task failed. Client %d return err status:" % i + str(client_resp)
            logger.logE(err)
            err_status = True
    if err_status:
        return resp_msg("err", client_resps)

    json.dump(post_data, open(ServerTaskRoot + task_name + ".json", "w+"), indent=4)
    logger.log("Task Created, Save json file in " + ServerTaskRoot + task_name + ".json")
    if data_json is not None:
        json.dump(data_json, open(ServerTaskRoot + "Data/" + task_name + ".json", "w+"), indent=4)
        logger.log("Distributed dataset initialized, saved in" + ServerTaskRoot + "Data/" + task_name + ".json")
    return resp_msg()


task_query_dict = dict()
http_query_dict = dict()


@main_server.route("/startTask", methods=["GET"])
def start_task():
    task_name = request.args.get("task_name")
    try:
        task_json = json.load(open(ServerTaskRoot + task_name + ".json", "r"))
    except Exception as e:
        err = "Load task error: {}".format(e)
        logger.logE("Cannot load task:\n" + err)
        return resp_msg("err", err)

    client_paras, client_addrs = generate_task_paras(task_json)

    client_errs, client_resps = broadcast_request([ClientProtocol + client_addr + "/startTask" for client_addr in client_addrs],
                                                  "get", params=[{"task_name": task_name, "client_id": i}
                                                                for i in range(len(client_addrs))])

    if len(client_errs) != 0:
        err = "Failed to startTask due to GET error: " + str(client_errs)
        logger.logE(err)
        return resp_msg("err", err)

    err_status = False
    for i, client_resp in enumerate(client_resps):
        if client_resp["status"] != "ok":
            err = "StartTask Failed. Client %d return err status:" % i + str(client_resp)
            logger.logE(err)
            err_status = True
    if err_status:
        return resp_msg("err", "StartTask Failed: " + str(client_resps))

    grpc_clients = [
        TaskQueryClient(client_addrs[i].split(":")[0] + ":%d" % client_para["client_config"]["computation_port"])
        for i, client_para in enumerate(client_paras)
    ]
    task_query_dict[task_name] = grpc_clients
    return resp_msg()


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
        return res
    except Exception as e:
        logger.logE("GRPC query failed: " + str(e))
        return resp_msg("err", "Query client grpc failed")


@main_server.route("/queryStatus", methods=["GET"])
def query_status():
    task_name = request.args.get("task_name")
    if task_name in http_query_dict:
        client_addrs = http_query_dict[task_name]
    else:
        if not os.path.isfile(ServerTaskRoot + task_name + ".json"):
            return resp_msg("ok", "NotExist")
        client_addrs = generate_task_paras(json.load(open(ServerTaskRoot + task_name + ".json")))[1]
        http_query_dict[task_name] = client_addrs
    try:
        resp = requests.get(ClientProtocol + client_addrs[0] + "/monitor",
                            params={"query": "task_status", "task_name": task_name, "client_id": 0})
    except:
        return resp_msg("Failed to get task status. The main client's address {} is unavailable".format(client_addrs[0]))
    if resp.status_code != requests.codes.ok:
        return resp_msg(
            "Failed to get task status. The main client's address {} is unavailable".format(client_addrs[0]))
    return resp.json()


@main_server.route("/queryDataset", methods=["GET"])
def query_dataset():
    task_name = request.args.get("task_name")
    if not os.path.isfile(ServerTaskRoot + "Data/" + task_name + ".json"):
        return resp_msg("err", "NotExist")

    data_dict = json.load(open(ServerTaskRoot + "Data/" + task_name + ".json"))
    try:
        addr = list(data_dict.keys())[0]
        resp = requests.get(ClientProtocol + addr + "/monitor", params={
            "query": "data_status",
            "data_name": data_dict[addr]
        })
        if resp.status_code != requests.codes.ok:
            return resp_msg("err", "HttpErrorCode: " + str(resp.status_code))
        return resp.json()
    except Exception as e:
        return resp_msg("err", "Error query client: " + str(e))


@main_server.route("/queryRecord", methods=["GET"])
def query_record():
    task_name = request.args.get("task_name")
    if task_name in http_query_dict:
        client_addrs = http_query_dict[task_name]
    else:
        if not os.path.isfile(ServerTaskRoot + task_name + ".json"):
            return resp_msg("ok", "NotExist")
        client_addrs = generate_task_paras(json.load(open(ServerTaskRoot + task_name + ".json")))[1]
        http_query_dict[task_name] = client_addrs
    try:
        resp = requests.get(ClientProtocol + client_addrs[-1] + "/monitor", params={
            "query": "record",
            "task_name": task_name,
            "client_id": len(client_addrs) - 1,
        })
        if resp.status_code != requests.codes.ok:
            return resp_msg("err", "HttpErrorCode: " + str(resp.status_code))
        return resp.json()
    except Exception as e:
        return resp_msg("err", "Error query client: " + str(e))