from flask import Flask, request
import requests
import json
import os
import sys
import traceback
from Utils.Log import Logger
import Server.TaskControl as TC
import threading


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

TaskDict = dict()


@client_server.route("/createTask", methods=["POST"])
def create_task():
    post_data = request.get_data(as_text=True)
    logger.log("Received createTask request with data:\n" + post_data)
    try:
        post_json = json.loads(post_data)
        post_json["log_config"] = {"log_level": ClientConfig.log_level}
    except Exception as e:
        return resp_msg("Error while parsing task parameters. Error:\n" + str(e) +
                        "\nTraceback:\n" + traceback.format_exc())

    try:
        TC.create_task_pyscript(**post_json)
    except Exception as e:
        return resp_msg("Error while creating task script. Error:\n" + str(e) +
                        "\nTraceback:\n" + traceback.format_exc())

    TaskDict[post_json["task_name"]] = post_json["listen_port"]
    return resp_msg()


@client_server.route("/startTask", methods=["GET"])
def start_task():
    task_name = request.args.get("task_name")
    client_id = request.args.get("client_id")
    task_path = TC.Config.TaskRootPath + task_name + "-" + client_id + "/"
    if os.path.isdir(task_path):
        def start_thread():
            lock_file = open(task_path + "run.lock", "w+")
            lock_file.close()
            os.system("python " + task_path + "task.py")
            os.remove(task_path + "run.lock")
        if os.path.isfile(task_path + "run.lock"):
            return resp_msg("err", "Task is running")
        threading.Thread(target=start_thread).start()
        return resp_msg()
    return resp_msg("ok", "Task is not created")