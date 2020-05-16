from Communication.RPCComm import Peer
from Client.DataProviders import DataClient, TripletsProvider
from Client.ComputationProviders import MainTFClient
from Client.Protocols import matrix_multiply_2pc
from Utils.Log import Logger
from Client.Data import RandomDataLoader

def test_mpc_start():
    ip_dict = {
        0: "127.0.0.1:19001",
        1: "127.0.0.1:19002",
        2: "127.0.0.1:19003",
        3: "127.0.0.1:19004",
        4: "127.0.0.1:19905"
    }
    channel0 = Peer(0, "[::]:19001", 10, ip_dict)
    channel1 = Peer(1, "[::]:19002", 10, ip_dict)
    channel2 = Peer(2, "[::]:19003", 10, ip_dict)
    channel3 = Peer(2, "[::]:19004", 10, ip_dict)
    main_client = MainTFClient(0, channel0, [2, 3], 4, logger=Logger(prefix="Main client:"))
    triplets_provider = TripletsProvider(1, channel1, logger=Logger(prefix="Triplet provider:"))
    data_client0 = DataClient(2, channel2, )

test_mm_2pc()