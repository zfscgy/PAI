from flask import Flask, request
import requests
import json
import traceback
from Utils.Log import Logger
import Server.TaskControl as TC



client_server = Flask(__name__)


class ClientConfig:

    server_log_path = "Test/Log/"

    log_level = 0

def resp_msg(status="ok", msg=None):
    return {
        "status": status,
        "msg": msg
    }


logger = Logger(open(ClientConfig.server_log_path + "/main_server-%d.log" % id(client_server), "a+"), level=0)

TaskDict = dict()


@client_server.route("/createTask")
def create_task():
    post_data = request.get_data(as_text=True)
    logger.log("Received createTask request with data:\n" + post_data)
    try:
        post_json = json.loads(post_data)
        post_json["log_config"] = { "log_level": ClientConfig.log_level }
    except Exception as e:
        return resp_msg("Error while parsing task parameters. Error:\n" + str(e) +
                        "\nTraceback:\n" + traceback.format_exc())

    try:
        task = TC.create_task(**post_json)
    except:
        return resp_msg("Error while creating task. Error:\n" + str(e) +
                        "\nTraceback:\n" + traceback.format_exc())

    TaskDict[task] = task
    return resp_msg()