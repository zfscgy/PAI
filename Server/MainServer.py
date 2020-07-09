import threading

from flask import Flask, request, jsonify
import requests
import json
import traceback
from Utils.Log import Logger


main_server = Flask(__name__)

class MainConfig:

    server_log_path = "Test/Log/"

    #
    learning_rate = 0.1
    batch_size = 64
    learn_config = {
        "loss": "mse",
        "metrics": "auc_ks"
    }


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
        task_name = post_data["task-name"]
        train_config = post_data["train-config"]

        # Generate IP dict
        clients = post_data["clients"]
        main_client = clients["main"]
        crypto_producer = clients["crypto-producer"]
        data_clients = clients["data-clients"]
        label_client = clients["label-client"]
        ip_dict = dict()
        ip_dict[0] = main_client["addr"]
        ip_dict[1] = crypto_producer["addr"]
        for i, data_client in enumerate(data_clients):
            ip_dict[2 + i] = data_client["addr"] + ":" + str(data_client["computation_port"])
        ip_dict[2 + len(data_clients)] = label_client["addr"]

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
            "lable_client": label_client_id,
            "main_client": main_client_id,
            "crypto_provider": crypto_producer_id,
            "ip_dict": ip_dict,
        }

        main_task_params = {
            "role": "main-client",
            "client_id": 0,
            "listen_port": main_client["computation_port"],
            "train_config": train_config,
        }

        crypto_producer_task_params = {
            "role": "crypto-producer",
            "client_id": 1,
            "listen_port": crypto_producer["computation_port"]
        }

        data_client_task_params = [{
            "role": "data-client",
            "client_id": c,
            "listen_port": c["computation_port"],
        } for c in data_clients]

        label_client_task_params = {
            "role": "label-client",
            "client_id": label_client_id,
            "learn_config": MainConfig.learn_config
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
            resp = requests.post("http://" + client["addr"] + ":" + str(client["http_port"]), json=post_data)
        except Exception as e:
            logger.logE("Post to client /createTask failed, error {}".format(e))
            post_errors_ref[client_id] = True
            return
        if resp.status_code != requests.codes.ok:
            logger.logE("Post to client /createTask failed, with status code %d" % resp.status_code)
            post_errors_ref[client_id] = True
        else:
            resp_msg = json.loads(resp.text)
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
        return resp_msg("err", "Failed to post to client /createTask, check the main server log. "
                               "Error state of clients:" + str(post_errors_dict))

    return resp_msg()
