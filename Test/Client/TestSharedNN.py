import threading
import time
import numpy as np
from Client.Learning.Metrics import AUC_KS
from Client.Learning.Losses import MSELoss

from Communication.RPCComm import Peer
from Client.MPCClient import MPCClientParas, MPCClientMode
from Client.SharedNN.DataProviders import FeatureClient, LabelClient
from Client.MPCProviders.TripletProducer import TripletProducer
from Client.SharedNN.ComputationProviders import MainClient
from Utils.Log import Logger
from Client.Data.DataLoader import CSVDataLoader


def test_credit_data_2pc():
    # Disable GPU since server do not have any computation other than sigmoid
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')

    print("\n======== Test mpc NN protocol with Credit Default TestDataset ============\n")
    ip_dict = {
        0: "127.0.0.1:19001",
        1: "127.0.0.1:19002",
        2: "127.0.0.1:19003",
        3: "127.0.0.1:19004",
        4: "127.0.0.1:19005"
    }
    channel0 = Peer(0, "[::]:19001", 10, ip_dict, 13, logger=Logger(prefix="Channel0:"))
    channel1 = Peer(1, "[::]:19002", 10, ip_dict, 13, logger=Logger(prefix="Channel1:", level=1))
    channel2 = Peer(2, "[::]:19003", 10, ip_dict, 13, logger=Logger(prefix="Channel2:"))
    channel3 = Peer(3, "[::]:19004", 10, ip_dict, 13, logger=Logger(prefix="Channel3:"))
    channel4 = Peer(4, "[::]:19005", 10, ip_dict, 13, logger=Logger(prefix="Channel4:"))
    train_config = {
        "client_dims": {2: 30, 3: 42},
        "out_dim": 1,
        "layers": [],
        "batch_size": 256,
        "test_per_batch": 101,
        "test_batch_size": None,
        "learning_rate": 0.1,
        "max_iter": 1002,
    }
    mpc_paras = MPCClientParas([2, 3], 4, 0, 1)
    main_client = MainClient(channel0, Logger(prefix="Main client:"), mpc_paras, MPCClientMode.Train, train_config)
    triplets_provider = TripletProducer(channel1, Logger(prefix="Triplet provider:"), mpc_paras)
    data_client0 = FeatureClient(channel2, Logger(prefix="Data client 0:"), mpc_paras, MPCClientMode.Train,
                                 CSVDataLoader("Test/TestDataset/Data/credit_default.csv", list(range(40000)), list(range(30))),
                                 CSVDataLoader("Test/TestDataset/Data/credit_default.csv", list(range(40000, 50000)), list(range(30))))
    data_client1 = FeatureClient(channel3, Logger(prefix="Data client 1:"), mpc_paras, MPCClientMode.Train,
                                 CSVDataLoader("Test/TestDataset/Data/credit_default.csv", list(range(40000)), list(range(30, 72))),
                                 CSVDataLoader("Test/TestDataset/Data/credit_default.csv", list(range(40000, 50000)), list(range(30, 72))))

    label_client = LabelClient(channel4, Logger(prefix="Lable client:"), mpc_paras, MPCClientMode.Train,
                               CSVDataLoader("Test/TestDataset/Data/credit_default.csv", list(range(40000)), list(range(72, 73))),
                               CSVDataLoader("Test/TestDataset/Data/credit_default.csv", list(range(40000, 50000)),
                                             list(range(72, 73))), MSELoss(), AUC_KS)
    triplets_provider.start_listening()

    main_client_start_th = threading.Thread(
        target=main_client.start_train,
    )
    data_client0_th = threading.Thread(target=data_client0.start_train)
    data_client1_th = threading.Thread(target=data_client1.start_train)
    label_client_th = threading.Thread(target=label_client.start_train)
    triplets_provider.start_listening()
    data_client0_th.start()
    data_client1_th.start()
    label_client_th.start()
    time.sleep(1)
    main_client_start_th.start()
    print("====== Stop the triplet provider, the training should be auto exited =========")
    main_client_start_th.join()
    data_client0_th.join()
    data_client1_th.join()
    label_client_th.join()
    print("====== MPC NN Test finished =============")
    np.savetxt("mpc_record.csv", np.array(label_client.metrics_record), delimiter=",")
    triplets_provider.stop_listening()


if __name__ == "__main__":
    test_credit_data_2pc()
