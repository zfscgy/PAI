from Server.TaskControl.MPCTask import MPCRoles, MPCModels

role_dict = {
    "main_client": MPCRoles.MainProvider,
    "crypto_producer": MPCRoles.CryptoProvider,
    "feature_clients": MPCRoles.FeatureProvider,
    "label_client": MPCRoles.LabelProvider
}


class TaskParaGenerator:
    def __init__(self, para_dict: dict):
        self.para_dict = para_dict
        self.ip_dict = None
        self.mpc_paras = None
        self.client_paras = None

    def generate_paras(self):
        para_dict = self.para_dict
        task_name = para_dict.get("task_name")
        if task_name is None:
            return "Para json must have key task_name"
        self.task_name = task_name

        clients = para_dict.get("clients")
        if clients is None:
            return "Para json must have key clients"
        try:
            mpc_paras = {
                "main_client_id": 0,
                "crypto_producer_id": 1,
                "feature_client_ids": [2 + i for i in range(len(clients["feature_clients"]))],
                "label_client_id": 2 + len(clients["feature_clients"])
            }
        except:
            return "Cannot generate mpc_paras. Para json is not correct."

        self.mpc_paras = mpc_paras
        try:
            all_clients = [clients["main_client"], clients["crypto_producer"]] + clients["feature_clients"] + clients["data_client"]
            all_roles = [MPCRoles.MainProvider, MPCRoles.CryptoProvider] + \
                        [MPCRoles.FeatureProvider] * len(clients["feature_clients"]) + [MPCRoles.LabelProvider]
        except:
            return "all_clients generate failed."

        self.all_clients = all_clients
        self.client_paras = [None for _ in range(len(all_clients))]
        self.ip_dict = dict()
        self.http_dict = dict()
        for i, client in enumerate(all_clients):
            self.ip_dict[i] = client["addr"] + ":%d" % client["computation_port"]
            self.http_dict[i] = client["addr"] + ":%d" % client["http_port"]
            self.client_paras[i] = {
                "client_id": i,
                "role": all_roles[i],
                "listen_port": client["computation_port"]
            }

        model_name = para_dict.get("model_name")
        if model_name is None:
            return "Para dict must have key model_name"
        self.model_name = model_name
        return  ""


def shared_mpc_para_generate(generator: TaskParaGenerator):
    """
    Generate additional parameters for shared_nn tasks
    :param generator:
    :return:
    """
    client_dims = dict()
    for i, client in enumerate(generator.all_clients[2:-1]):
        client_dims[i] = client["dim"]

    if generator.model_name is MPCModels.SharedLR:
        out_dim = 1
        layers = []
    else:
        out_dim = 64
        layers = [1]

    generator.client_paras[0]["configs"] = {
        "client_dims": client_dims,
        "out_dim": out_dim,
        "layers": layers,
        "batch_size": 64,
        "test_per_batch": 1001,
        "test_batch_size": None,
        "max_iter": generator.para_dict["train_config"]["max_iter"],
    }
    generator.client_paras[-1]["configs"] = {
        "loss_func": generator["train_config"]["loss"],
        "metrics": generator["train_config"]["metrics"]
    }


generator_dict = {
    MPCModels.SharedNN: shared_mpc_para_generate,
    MPCModels.SharedLR: shared_mpc_para_generate
}


def generate_paras(para_dict: dict):
    task_para_generator = TaskParaGenerator(para_dict)
    err = task_para_generator.generate_paras()
    if err != "":
        return err
    if task_para_generator.model_name not in generator_dict:
        return "Model name not exist"
    else:
        try:
            generator_dict[task_para_generator.model_name](task_para_generator)
        except Exception as e:
            return "Generate model {}'s para error: {}".format(task_para_generator.model_name, err)
    return task_para_generator