from flask import Flask, request
import json
import os
from Utils.Log import Logger
import Task.TaskScriptMaker as TM
import Task.TaskConfig as Config
import threading
import Task.Utils as Utils
import Task.Monitor as monitor


client_server = Flask(__name__)


class ClientConfig:
    server_log_path = "Test/Log/"
    log_level = 0


def resp_msg(status="ok", msg=None):
    return {
        "status": status,
        "msg": msg
    }


logger = Logger(open(ClientConfig.server_log_path + "/client_server-%d_log.txt" % id(client_server), "a+"), level=0)


@client_server.route("/createTask", methods=["POST"])
def create_task():
    post_data = request.get_data(as_text=True)
    logger.log("Received createTask request with data:\n" + post_data)
    try:
        post_json = json.loads(post_data)
    except Exception as e:
        err = "Error while parsing task parameters. Error:\n" + str(e)
        logger.logE(err)
        return resp_msg("err", err)
    try:
        TM.create_task_pyscript(**post_json)
    except Exception as e:
        err = "Error while creating task script. Error:\n" + str(e)
        logger.logE(err)
        return resp_msg("err", err)
    return resp_msg()


@client_server.route("/startTask", methods=["GET"])
def start_task():
    task_name = request.args.get("task_name")
    client_id = request.args.get("client_id")
    task_path = Config.TaskRootPath + task_name + "-" + client_id + "/"
    if os.path.isdir(task_path):
        def start_thread():
            lock_file = open(task_path + "run.lock", "w+")
            lock_file.close()
            os.system("python " + task_path + "task.py")
            os.remove(task_path + "run.lock")
        if os.path.isfile(task_path + "run.lock"):
            return resp_msg("err", "Task is already running")
        if os.path.isfile(task_path + "finished"):
            return resp_msg("err", "Task is already finished")
        if os.path.isfile(task_path + "failed"):
            return resp_msg("err", "Task is already failed")
        threading.Thread(target=start_thread).start()
        return resp_msg()
    return resp_msg("err", "Task is not created")


@client_server.route("/monitor", methods=["GET"])
def query_monitor():
    monitor_dict = {
        "task_status": (monitor.task_status, ["task_name", "client_id"], [str, int]),
        "data_status": (monitor.data_status, ["data_name"], [str]),
        "record": (monitor.train_result, ["task_name", "client_id"], [str, int])
    }
    query = request.args.get("query")
    if query is None:
        return resp_msg("err", "Invalid query")
    params = monitor_dict[query][1]
    types = monitor_dict[query][2]
    kwargs = dict()
    for i, param in enumerate(params):
        if param not in request.args:
            return resp_msg("err", "Param {} unfilled".format(param))
        kwargs[param] = request.args.get(param, type=types[i])
    try:
        return monitor_dict[query][0](**kwargs)
    except Exception as e:
        logger.logE("Error while query monitor: " + str(e))
        return resp_msg("err", str(e))
