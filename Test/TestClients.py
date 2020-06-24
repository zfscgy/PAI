import threading
import time
from sklearn.metrics import roc_auc_score


from Communication.RPCComm import Peer
from Client.DataProviders import DataClient, LabelClient
from Client.SMCProvider import TripletsProvider
from Client.ComputationProviders import MainTFClient
from Utils.Log import Logger
from Client.Data import RandomDataLoader, CSVDataLoader


def test_mpc():
    print("\n======== Test mpc NN protocol ============\n")
    ip_dict = {
        0: "127.0.0.1:19001",
        1: "127.0.0.1:19002",
        2: "127.0.0.1:19003",
        3: "127.0.0.1:19004",
        4: "127.0.0.1:19005"
    }
    channel0 = Peer(0, "[::]:19001", 10, ip_dict, 3, logger=Logger(prefix="Channel0:"))
    channel1 = Peer(1, "[::]:19002", 10, ip_dict, 3, logger=Logger(prefix="Channel1:", level=1))
    channel2 = Peer(2, "[::]:19003", 10, ip_dict, 3, logger=Logger(prefix="Channel2:"))
    channel3 = Peer(3, "[::]:19004", 10, ip_dict, 3, logger=Logger(prefix="Channel3:"))
    channel4 = Peer(4, "[::]:19005", 10, ip_dict, 3, logger=Logger(prefix="Channel4:"))
    main_client = MainTFClient(channel0, [2, 3], 4, logger=Logger(prefix="Main client:"))
    triplets_provider = TripletsProvider(channel1, logger=Logger(prefix="Triplet provider:"))
    data_client0 = DataClient(channel2, RandomDataLoader(10), server_id=0, triplets_id=1, other_data_clients=[3],
                              logger=Logger(prefix="Data client 0:"))
    data_client1 = DataClient(channel3, RandomDataLoader(20), server_id=0, triplets_id=1, other_data_clients=[2],
                              logger=Logger(prefix="Data client 1:"))
    label_client = LabelClient(channel4, RandomDataLoader(1), server_id=0, logger=Logger(prefix="Lable client:"))
    triplets_provider.start_listening()
    data_client0_th = threading.Thread(target=data_client0.start_train)
    data_client1_th = threading.Thread(target=data_client1.start_train)
    label_client_th = threading.Thread(target=label_client.start_train)
    main_client_send_config_th = threading.Thread(
        target=main_client.send_config_message,
        args=({
                  "client_dims": {2: 10, 3: 20},
                  "out_dim": 10,
                  "batch_size": 10,
                  "learning_rate": 0.01
              },)
    )

    data_client0_th.start()
    data_client1_th.start()
    label_client_th.start()
    main_client.build_default_network(10, 1)
    time.sleep(15)
    main_client_send_config_th.start()

    main_client_send_config_th.join()
    triplets_provider.start_listening()
    time.sleep(0.5)
    print("====== Configuration message sent =========")
    main_client_start_th = threading.Thread(target=main_client.start_train)
    main_client_start_th.start()
    print("====== Stop the triplet provider, the training should be auto exited =========")
    time.sleep(20)
    triplets_provider.stop_listening()
    main_client_start_th.join()
    data_client0_th.join()
    data_client1_th.join()
    label_client_th.join()

    print("====== MPC NN Test finished =============")


def test_2pc_mnist():
    print("\n======== Test mpc NN protocol with MNIST Dataset ============\n")
    ip_dict = {
        0: "127.0.0.1:19001",
        1: "127.0.0.1:19002",
        2: "127.0.0.1:19003",
        3: "127.0.0.1:19004",
        4: "127.0.0.1:19005"
    }
    channel0 = Peer(0, "[::]:19001", 10, ip_dict, 3, logger=Logger(prefix="Channel0:"))
    channel1 = Peer(1, "[::]:19002", 10, ip_dict, 3, logger=Logger(prefix="Channel1:", level=1))
    channel2 = Peer(2, "[::]:19003", 10, ip_dict, 3, logger=Logger(prefix="Channel2:"))
    channel3 = Peer(3, "[::]:19004", 10, ip_dict, 3, logger=Logger(prefix="Channel3:"))
    channel4 = Peer(4, "[::]:19005", 10, ip_dict, 3, logger=Logger(prefix="Channel4:"))
    main_client = MainTFClient(channel0, [2, 3], 4, logger=Logger(prefix="Main client:"))
    triplets_provider = TripletsProvider(channel1, logger=Logger(prefix="Triplet provider:"))
    data_client0 = DataClient(channel2,
                              CSVDataLoader("Test/TestDataset/mnist.csv", list(range(50000)), list(range(300))),
                              CSVDataLoader("Test/TestDataset/mnist.csv", list(range(50000, 55000)), list(range(300))),
                              server_id=0, triplets_id=1, other_data_clients=[3],
                              logger=Logger(prefix="Data client 0:"))
    data_client1 = DataClient(channel3,
                              CSVDataLoader("Test/TestDataset/mnist.csv", list(range(50000)), list(range(300, 784))),
                              CSVDataLoader("Test/TestDataset/mnist.csv", list(range(50000, 55000)),
                                            list(range(300, 784))),
                              server_id=0, triplets_id=1, other_data_clients=[2],
                              logger=Logger(prefix="Data client 1:"))
    label_client = LabelClient(channel4,
                               CSVDataLoader("Test/TestDataset/mnist.csv", list(range(50000)), list(range(784, 794))),
                               CSVDataLoader("Test/TestDataset/mnist.csv", list(range(50000, 55000)),
                                             list(range(784, 794))),
                               server_id=0, logger=Logger(prefix="Lable client:"))
    triplets_provider.start_listening()
    data_client0_th = threading.Thread(target=data_client0.start_train)
    data_client1_th = threading.Thread(target=data_client1.start_train)
    label_client_th = threading.Thread(target=label_client.start_train)
    config = {
        "client_dims": {2: 300, 3: 484},
        "out_dim": 150,
        "batch_size": 32,
        "test_per_batch": 100,
        "test_batch_size": 1000,
        "learning_rate": 0.01,
        "sync_info": {
            "seed": 8964
        }
    }
    main_client.set_config_message(config)
    main_client_send_config_th = threading.Thread(
        target=main_client.send_config_message,
        args=(config,)
    )

    data_client0_th.start()
    data_client1_th.start()
    label_client_th.start()
    main_client.build_default_network(150, 10)
    time.sleep(15)
    main_client_send_config_th.start()

    main_client_send_config_th.join()
    triplets_provider.start_listening()
    time.sleep(0.5)
    print("====== Configuration message sent =========")
    main_client_start_th = threading.Thread(target=main_client.start_train)
    main_client_start_th.start()
    print("====== Stop the triplet provider, the training should be auto exited =========")
    time.sleep(200)
    triplets_provider.stop_listening()
    main_client_start_th.join()
    data_client0_th.join()
    data_client1_th.join()
    label_client_th.join()

    print("====== MPC NN Test finished =============")

def test_credit_data_2pc():
    print("\n======== Test mpc NN protocol with MNIST Dataset ============\n")
    ip_dict = {
        0: "127.0.0.1:19001",
        1: "127.0.0.1:19002",
        2: "127.0.0.1:19003",
        3: "127.0.0.1:19004",
        4: "127.0.0.1:19005"
    }
    channel0 = Peer(0, "[::]:19001", 10, ip_dict, 3, logger=Logger(prefix="Channel0:"))
    channel1 = Peer(1, "[::]:19002", 10, ip_dict, 3, logger=Logger(prefix="Channel1:", level=1))
    channel2 = Peer(2, "[::]:19003", 10, ip_dict, 3, logger=Logger(prefix="Channel2:"))
    channel3 = Peer(3, "[::]:19004", 10, ip_dict, 3, logger=Logger(prefix="Channel3:"))
    channel4 = Peer(4, "[::]:19005", 10, ip_dict, 3, logger=Logger(prefix="Channel4:"))

    main_client = MainTFClient(channel0, [2, 3], 4, logger=Logger(prefix="Main client:"))
    triplets_provider = TripletsProvider(channel1, logger=Logger(prefix="Triplet provider:"))
    data_client0 = DataClient(channel2,
                              CSVDataLoader("Test/TestDataset/credit_default.csv", list(range(40000)), list(range(30))),
                              CSVDataLoader("Test/TestDataset/credit_default.csv", list(range(40000, 50000)), list(range(30))),
                              server_id=0, triplets_id=1, other_data_clients=[3],
                              logger=Logger(prefix="Data client 0:"))
    data_client1 = DataClient(channel3,
                              CSVDataLoader("Test/TestDataset/credit_default.csv", list(range(40000)), list(range(30, 72))),
                              CSVDataLoader("Test/TestDataset/credit_default.csv", list(range(40000, 50000)),
                                            list(range(30, 72))),
                              server_id=0, triplets_id=1, other_data_clients=[2],
                              logger=Logger(prefix="Data client 1:"))

    def auc(y_true, y_pred):
        return roc_auc_score(y_true[:, 0], y_pred[:, 0])
    label_client = LabelClient(channel4,
                               CSVDataLoader("Test/TestDataset/credit_default.csv", list(range(40000)), list(range(72, 73))),
                               CSVDataLoader("Test/TestDataset/credit_default.csv", list(range(40000, 50000)),
                                             list(range(72, 73))),
                               server_id=0, metric_func=auc, logger=Logger(prefix="Lable client:"))
    triplets_provider.start_listening()
    config = {
        "client_dims": {2: 30, 3: 42},
        "out_dim": 1,
        "batch_size": 256,
        "test_per_batch": 100,
        "test_batch_size": None,
        "learning_rate": 0.01,
        "sync_info": {
            "seed": 8964
        }
    }
    main_client.build_mlp_network(1, [])


    main_client_start_th = threading.Thread(
        target=main_client.start_train,
        args=(config,)
    )
    data_client0_th = threading.Thread(target=data_client0.start_train)
    data_client1_th = threading.Thread(target=data_client1.start_train)
    label_client_th = threading.Thread(target=label_client.start_train)
    data_client0_th.start()
    data_client1_th.start()
    label_client_th.start()
    main_client_start_th.start()
    triplets_provider.start_listening()
    print("====== Stop the triplet provider, the training should be auto exited =========")
    time.sleep(2000)
    triplets_provider.stop_listening()
    main_client_start_th.join()
    data_client0_th.join()
    data_client1_th.join()
    label_client_th.join()

    print("====== MPC NN Test finished =============")


if __name__ == "__main__":
    test_credit_data_2pc()
