from Client.Learning.Metrics import AUC_KS
from Client.Learning.Losses import MSELoss

from Communication.RPCComm import Peer
from Client.MPCClient import MPCClientParas, ClientMode
from Client.SharedNN.DataProviders import LabelClient
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

channel = Peer(4, "[::]:19005", 3, ip_dict, 13, logger=Logger(prefix="Channel4:"))

label_client = LabelClient(channel, Logger(prefix="Lable client:"), mpc_paras, ClientMode.Train,
                           CSVDataLoader("Test/TestDataset/Data/credit_default.csv", list(range(40000)),
                                         list(range(72, 73))),
                           CSVDataLoader("Test/TestDataset/Data/credit_default.csv", list(range(40000, 50000)),
                                         list(range(72, 73))), MSELoss(), AUC_KS)

label_client.start_train()