import threading

from flask import Flask, request
import requests
import json
import traceback
import os
from Utils.Log import Logger


main_server = Flask(__name__)

from Server.MainServerConfig import MainConfig


logger = Logger(open(MainConfig.server_log_path + "/main_server-%d_log.txt" % id(main_server), "a"), level=0)


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

    try:
        task_name = post_data["task_name"]

        if os.path.isfile(MainConfig.server_task_root + task_name + ".json"):
            logger.log("Task with name {} already exists".format(task_name))
            return resp_msg("err", "Task with name {} already exists".format(task_name))

        train_config = post_data["train-config"]

        # Generate IP dict
        clients = post_data["clients"]
        main_client = clients["main_client"]
        crypto_producer = clients["crypto_producer"]
        data_clients = clients["data_clients"]
        label_client = clients["label_client"]
        ip_dict = dict()
        ip_dict[0] = main_client["addr"] + ":" + str(main_client["computation_port"])
        ip_dict[1] = crypto_producer["addr"] + ":" + str(main_client["computation_port"])
        for i, data_client in enumerate(data_clients):
            ip_dict[2 + i] = data_client["addr"] + ":" + str(data_client["computation_port"])
        ip_dict[2 + len(data_clients)] = label_client["addr"] + ":" + str(label_client["computation_port"])

        main_client_id = 0
        crypto_producer_id = 1
        data_client_ids = [2 + i for i in range(len(data_clients))]
        label_client_id = 2 + len(data_clients)

        if post_data["model-name"] is "nn":
            layers = [1]
            client_out_dim = 64
        else:
            layers = []
            client_out_dim = 1

        # build train config for main client
        dims = [client["dim"] for client in data_clients]
        train_config["client_dims"] = dict()
        for i, dim in enumerate(dims):
            train_config["client_dims"][i + 2] = dim
        train_config["out_dim"] = client_out_dim
        train_config["layers"] = layers
        train_config["learning_rate"] = MainConfig.learning_rate
        train_config["batch_size"] = MainConfig.batch_size

        common_params = {
            "task_name": task_name,
            "data_clients": data_client_ids,
            "label_client": label_client_id,
            "main_client": main_client_id,
            "crypto_provider": crypto_producer_id,
            "ip_dict": ip_dict,
        }

        main_task_params = {
            "role": "main_client",
            "client_id": 0,
            "listen_port": main_client["computation_port"],
            "train_config": train_config,
        }

        crypto_producer_task_params = {
            "role": "crypto_producer",
            "client_id": 1,
            "listen_port": crypto_producer["computation_port"]
        }

        data_client_task_params = [{
            "role": "data_client",
            "client_id": data_client_ids[i],
            "listen_port": c["computation_port"],
            "data_config": {"data_path": c["data-file"]}
        } for i, c in enumerate(data_clients)]

        label_client_task_params = {
            "role": "label_client",
            "client_id": label_client_id,
            "listen_port": label_client["computation_port"],
            "data_config": {"data_path": label_client["data-file"]},
            "learn_config": MainConfig.learn_config,
        }

    except Exception as e:
        logger.logE("Error while generating task parameters. Stop.")
        return resp_msg("err", "Generating task parameters failed. Error: \n" + str(e) +
                        "Stack trace:\n" + traceback.format_exc())

    post_errors = [False for _ in range(len(data_clients) + 3)]

    def post_to_client_webserver(client_id, post_errors_ref):
        global post_error
        if client_id == 0:
            post_data = main_task_params
            client = main_client
        elif client_id == 1:
            post_data = crypto_producer_task_params
            client = crypto_producer
        elif 1 < client_id < 2 + len(data_clients):
            post_data = data_client_task_params[client_id - 2]
            client = data_clients[client_id - 2]
        else:
            post_data = label_client_task_params
            client = label_client
        post_data.update(common_params)
        try:
            resp = requests.post(MainConfig.client_protocol + client["addr"] + ":" + str(client["http_port"]) + "/createTask", json=post_data)
        except Exception as e:
            logger.logE("Post to client /createTask failed, error {}".format(e))
            post_errors_ref[client_id] = True
            return
        if resp.status_code != requests.codes.ok:
            logger.logE("Post to client /createTask failed, with status code %d" % resp.status_code)
            post_errors_ref[client_id] = True
        else:
            resp_msg = resp.json()
            if resp_msg["status"] != "ok":
                logger.logE("Received error response from client {}, message {}".format(client_id, resp_msg))
                post_errors_ref[client_id] = True

    posting_threads = []
    for client_id in [main_client_id, crypto_producer_id, label_client_id] + data_client_ids:
        posting_threads.append(threading.Thread(target=post_to_client_webserver, args=(client_id, post_errors),
                                                name="Thread-PostCreateTask-to-Client-%d" % client_id))
        posting_threads[-1].start()

    for posting_thread in posting_threads:
        posting_thread.join()

    if True in post_errors:
        post_errors_dict = dict()
        for i, e in enumerate(post_errors):
            if e:
                post_errors_dict[i] = "Post failed"
            else:
                post_errors_dict[i] = "Post success"
        logger.logE("Post startTask to clients failed. Post error dict:\n" + str(post_errors_dict))
        return resp_msg("err", "Failed to post to client /createTask, check the main server log. "
                               "Error state of clients:" + str(post_errors_dict))
    json.dump(post_data, open(MainConfig.server_task_root + task_name + ".json", "w+"), indent=4)
    logger.log("Task Created, Save json file in " + MainConfig.server_task_root + task_name + ".json")
    return resp_msg()


@main_server.route("/startTask", methods=["GET"])
def start_task():
    task_name = request.args.get("task_name")
    try:
        task_json = json.load(open(MainConfig.server_task_root + task_name + ".json", "r"))
    except Exception as e:
        err_msg = "Load task error: {}".format(e)
        logger.logE("Cannot load task:\n" + err_msg)
        return resp_msg("err", err_msg)

    clients = task_json["clients"]

    addr_dict = dict()
    addr_keys = []

    main_client_addr = MainConfig.client_protocol + clients["main_client"]["addr"] + ":%d" % clients["main_client"]["http_port"]
    addr_dict[main_client_addr] = "main_client"
    addr_keys.append(main_client_addr)

    crypto_producer_addr = MainConfig.client_protocol + clients["crypto_producer"]["addr"] + ":%d" % clients["crypto_producer"]["http_port"]
    addr_dict[crypto_producer_addr] = "crypto_producer"
    addr_keys.append(crypto_producer_addr)

    data_client_addrs = []
    for data_client in clients["data_clients"]:
        data_client_addrs.append(MainConfig.client_protocol + data_client["addr"] + ":%d" % data_client["http_port"])
        addr_dict[data_client_addrs[-1]] = "data_client"
        addr_keys.append(data_client_addrs[-1])

    label_client_addr = MainConfig.client_protocol + clients["label_client"]["addr"] + ":%d" % clients["label_client"]["http_port"]
    addr_dict[label_client_addr] = "label_client"
    addr_keys.append(label_client_addr)

    startTask_errors = []
    def request_to_clients(client_id, client_addr, startTask_errors: list):
        try:
            resp = requests.get(client_addr + "/startTask", {"task_name": task_name, "client_id": client_id})
        except Exception as e:
            logger.logE("Get request for client startTask failed with client address, Error:\n" + str(e))
            startTask_errors.append((client_addr, str(e)))
            return
        if resp.status_code != requests.codes.ok:
            err_msg = "Get request for client startTask failed with status code {}, client address {}".\
                       format(resp.status_code, client_addr)
            logger.log(err_msg)
            startTask_errors.append((client_addr, err_msg))
            return
        resp = resp.json()
        if resp["status"] != "ok":
            logger.log("Get request for client startTask failed with error message {}, client address {}".
                       format(resp["msg"], client_addr))
            startTask_errors.append((client_addr, resp["msg"]))
            return

    request_tasks = []
    for i, addr in enumerate(addr_keys):
        request_tasks.append(threading.Thread(target=request_to_clients, args=(i, addr, startTask_errors)))
        request_tasks[-1].start()
    for request_task in request_tasks:
        request_task.join()

    if len(startTask_errors) == 0:
        return resp_msg()

    else:
        error_msgs = "Get request for client startTask failed with message:\n" + "\n".\
            join([addr + msg + '(' + addr_dict[addr] + ')' for addr, msg in startTask_errors])
        logger.log(error_msgs)
        return resp_msg("err", error_msgs)