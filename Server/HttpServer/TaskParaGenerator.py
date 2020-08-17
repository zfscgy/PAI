from Server.HttpServer.ServerConfig import ClientProtocol


class ParaGenerationException(Exception):
    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self):
        return self.msg


def generate_task_paras(paras: dict):
    clients = paras["clients"]
    client_configs = [client["client_config"].copy() for client in clients]
    client_http_addrs = [client["addr"] + ":" + str(client["http_port"]) for client in clients]
    client_computation_addrs = [client["addr"] + ":" + str(client["client_config"]["computation_port"])
                                for client in clients]

    ip_dict = dict()
    for i, computation_addr in enumerate(client_computation_addrs):
        ip_dict[i] = computation_addr

    mpc_paras = {
        "main_client_id": -1,
        "crypto_producer_id": -1,
        "feature_client_ids": [],
        "label_client_id": -1,
    }
    for i, client in enumerate(clients):
        if client["role"] == "feature_client":
            role = client["role"] + "_ids"
        else:
            role = client["role"] + "_id"
        if role in mpc_paras:
            if isinstance(mpc_paras[role], int):
                if mpc_paras[role] != -1:
                    raise ParaGenerationException("Role {} already set.".format(role))
                else:
                    mpc_paras[role] = i
            elif isinstance(mpc_paras[role], list):
                mpc_paras[role].append(i)
        else:
            raise ParaGenerationException("Unrecognized role: " + client["role"])

    for client_config in client_configs:
        client_config["mpc_paras"] = mpc_paras

    client_task_paras = []
    for i, client_config in enumerate(client_configs):
        client_task_paras.append(
            {
                "task_name": paras["task_name"],
                "client_id": i,
                "client_port": client_config["computation_port"],
                "ip_dict": ip_dict,
                "client_config": client_config
            }
        )

    return client_task_paras, client_http_addrs


def generate_dataset_json(paras: dict):
    data_dict = dict()
    for client in paras["clients"]:
        if "out_data_path" in client["client_config"]:
            data_dict[client["addr"] + ":%d" % client["http_port"]] = client["client_config"]["out_data_path"]
    if len(data_dict) == 0:
        return None
    else:
        return data_dict
