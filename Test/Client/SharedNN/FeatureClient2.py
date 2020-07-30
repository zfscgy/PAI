from Communication.RPCComm import Peer
from Client.MPCClient import MPCClientParas, ClientMode
from Client.SharedNN.DataProviders import FeatureClient
from Utils.Log import Logger
from Client.Data.DataLoader import CSVDataLoader

ip_dict = {
    0: "127.0.0.1:19001",
    1: "127.0.0.1:19002",
    2: "127.0.0.1:19003",
    3: "127.0.0.1:19004",
    4: "127.0.0.1:19005"
}
mpc_paras = MPCClientParas([2, 3], 4, 0, 1)
channel = Peer(3, "[::]:19004", 3, ip_dict, 13, logger=Logger(prefix="Channel3:"))
feature_client2 = FeatureClient(channel, Logger(prefix="Data client 1:"), mpc_paras, ClientMode.Train,
                             CSVDataLoader("Test/TestDataset/Data/credit_default.csv", list(range(40000)),
                                           list(range(30, 72))),
                             CSVDataLoader("Test/TestDataset/Data/credit_default.csv", list(range(40000, 50000)),
                                           list(range(30, 72))))
feature_client2.start_train()