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

ip_dict = {
    0: "127.0.0.1:19001",
    1: "127.0.0.1:19002",
    2: "127.0.0.1:19003",
    3: "127.0.0.1:19004",
    4: "127.0.0.1:19005"
}

channel = Peer(1, "[::]:19002", 3, ip_dict, 13, logger=Logger(prefix="Channel1:", level=1))

mpc_paras = MPCClientParas([2, 3], 4, 0, 1)

triplets_provider = TripletProducer(channel, Logger(prefix="Triplet provider:"), mpc_paras, [2, 3])

triplets_provider.start_listening()