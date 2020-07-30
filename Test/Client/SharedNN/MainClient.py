from Communication.RPCComm import Peer
from Client.MPCClient import MPCClientParas, ClientMode
from Client.SharedNN.ComputationProviders import MainClient
from Utils.Log import Logger

ip_dict = {
    0: "127.0.0.1:19001",
    1: "127.0.0.1:19002",
    2: "127.0.0.1:19003",
    3: "127.0.0.1:19004",
    4: "127.0.0.1:19005"
}
channel = Peer(0, "[::]:19001", 3, ip_dict, 13, logger=Logger(prefix="Channel0:"))

mpc_paras = MPCClientParas([2, 3], 4, 0, 1)
main_client = MainClient(channel, Logger(prefix="Main client:"), mpc_paras, ClientMode.Train,
                         in_dim=32, out_dim=1, layers=[1], batch_size=64, test_batch_size=10000,
                         test_per_batches=101, learning_rate=0.1, max_iter=1000)

main_client.start_train()