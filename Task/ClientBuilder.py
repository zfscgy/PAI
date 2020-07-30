from Client.MPCProviders.TripletProducer import TripletProducer
from Client.SharedNN.ComputationProviders import MainClient as SharedNN_MainClient
from Client.SharedNN.DataProviders import FeatureClient as SharedNN_FeatureClient
from Client.SharedNN.DataProviders import LabelClient as SharedNN_LabelClient
from Client.Preprocess.AlignmentClient import PreprocessClient as Alignment_DataClient
from Client.Preprocess.AlignmentClient import MainPreprocessor as Alignment_MainClient

from Client.Data.DataLoader import CSVDataLoader
from Client.Learning.Losses import get_loss
from Client.Learning.Metrics import get_metric


class ClientHandle:
    def __init__(self, client, calls: dict, start):
        self.client = client
        self.calls = calls
        self.start = start


def build_TripletProducer(arg_dict: dict):
    client = TripletProducer(arg_dict["channel"], arg_dict["logger"], arg_dict["mpc_paras"], arg_dict["listen_clients"])
    return ClientHandle(client, dict(), client.start_listening)


def build_SharedNN_MainClient(arg_dict: dict):
    client = SharedNN_MainClient(arg_dict["channel"], arg_dict["logger"], arg_dict["mpc_paras"],
                                 arg_dict["in_dim"], arg_dict["out_dim"], arg_dict["layers"],
                                 arg_dict.get("batch_size") or 64,
                                 arg_dict.get("test_batch_size") or 10000,
                                 arg_dict.get("test_per_batches") or 1001,
                                 arg_dict.get("learning_rate") or 0.1,
                                 arg_dict.get("max_iter") or 1001)
    return ClientHandle(client, {"n_batches": lambda: client.n_rounds}, client.start_train)


def build_SharedNN_FeatureClient(arg_dict: dict):
    client = SharedNN_FeatureClient(arg_dict["channel"], arg_dict["logger"], arg_dict["mpc_paras"],
                                    CSVDataLoader(arg_dict["data_path"] + "train.csv"),
                                    CSVDataLoader(arg_dict["data_path"] + "test.csv"))
    return ClientHandle(client, dict(), client.start_train)


def build_SharedNN_LabelClient(arg_dict: dict):
    client = SharedNN_LabelClient(arg_dict["channel"], arg_dict["logger"], arg_dict["mpc_paras"],
                                  CSVDataLoader(arg_dict["data_path"] + "train.csv"),
                                  CSVDataLoader(arg_dict["data_path"] + "test.csv"),
                                  get_loss(arg_dict["loss"]),
                                  get_metric(arg_dict["metric"]),
                                  arg_dict["task_path"])
    return ClientHandle(client, {"record": lambda: client.test_record}, client.start_train)


def build_Alignment_DataClient(arg_dict: dict):
    client = Alignment_DataClient(arg_dict["channel"], arg_dict["logger"], arg_dict["mpc_paras"],
                                  arg_dict["raw_data_path"], arg_dict["out_data_path"])
    return ClientHandle(client, dict(), client.start_align)


def build_Alignment_MainClient(arg_dict: dict):
    client = Alignment_MainClient(arg_dict["channel"], arg_dict["logger"], arg_dict["mpc_paras"])
    return ClientHandle(client, dict(), client.start_align)


client_builder_dict = {
    "triplet_producer": build_TripletProducer,
    "shared_nn_feature": build_SharedNN_FeatureClient,
    "shared_nn_label": build_SharedNN_LabelClient,
    "shared_nn_main": build_SharedNN_MainClient,
    "alignment_data": build_Alignment_DataClient,
    "alignment_main": build_Alignment_MainClient
}


def build_client(arg_dict: dict):
    client_type = arg_dict["client_type"]
    return client_builder_dict[client_type](arg_dict)
