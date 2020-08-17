import threading
import time
import numpy as np
from Client.Learning.Metrics import AUC_KS
from Client.Learning.Losses import MSELoss


from Communication.RPCComm import Peer
from Client.MPCClient import MPCClientParas, ClientMode
from Client.SharedNN.DataProviders import FeatureClient, LabelClient
from Client.MPCProviders.TripletProducer import TripletProducer
from Client.SharedNN.ComputationProviders import MainClient
from Utils.Log import Logger
from Client.Data.DataLoader import CSVDataLoader


def test_credit_data_2pc():
    # Disable GPU since server do not have any computation other than sigmoid

    print("\n======== Test mpc SharedNN protocol with Credit Default TestDataset ============\n")
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
    mpc_paras = MPCClientParas([2, 3], 4, 0, 1)
    main_client = MainClient(channel0, Logger(prefix="Main client:"), mpc_paras,
                             in_dim=64, out_dim=1, layers=[1], batch_size=64, test_batch_size=10000,
                             test_per_batches=11, learning_rate=0.1, max_iter=33)
    triplets_provider = TripletProducer(channel1, Logger(prefix="Triplet provider:"), mpc_paras, [2, 3])
    data_client0 = FeatureClient(channel2, Logger(prefix="Data client 0:"), mpc_paras,
                                 CSVDataLoader("Test/TestDataset/Data/credit_default.csv", list(range(40000)), list(range(30))),
                                 CSVDataLoader("Test/TestDataset/Data/credit_default.csv", list(range(40000, 50000)), list(range(30))))
    data_client1 = FeatureClient(channel3, Logger(prefix="Data client 1:"), mpc_paras,
                                 CSVDataLoader("Test/TestDataset/Data/credit_default.csv", list(range(40000)), list(range(30, 72))),
                                 CSVDataLoader("Test/TestDataset/Data/credit_default.csv", list(range(40000, 50000)), list(range(30, 72))))

    label_client = LabelClient(channel4, Logger(prefix="Lable client:"), mpc_paras,
                               CSVDataLoader("Test/TestDataset/Data/credit_default.csv", list(range(40000)), list(range(72, 73))),
                               CSVDataLoader("Test/TestDataset/Data/credit_default.csv", list(range(40000, 50000)),
                                             list(range(72, 73))), MSELoss(), AUC_KS, "")
    main_client_start_th = threading.Thread(
        target=main_client.start_train,
    )
    data_client0_th = threading.Thread(target=data_client0.start_train)
    data_client1_th = threading.Thread(target=data_client1.start_train)
    label_client_th = threading.Thread(target=label_client.start_train)
    triplets_provider_th = threading.Thread(target=triplets_provider.start_listening)
    triplets_provider_th.start()
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
    print("====== MPC SharedNN Test finished =============")
    np.savetxt("mpc_record.csv", np.array(label_client.test_record), delimiter=",")
    triplets_provider.stop_listening()


if __name__ == "__main__":
    test_credit_data_2pc()
