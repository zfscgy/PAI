from Communication.RPCComm import Peer
from Client.DataProviders import DataClient, TripletsProvider
from Client.Protocols import matrix_multiply_2pc

def test_mm_2pc():
    ip_dict = {
        0: "127.0.0.1:19001",
        1: "127.0.0.1:19002",
        2: "127.0.0.1:19003"
    }
    channel0 = Peer(0, "[::]:19001", 10, ip_dict)
    client0 = DataClient(0, channel0, 32, 64, 16, triplets_id=2)
    channel1 = Peer(1, "[::]:19002", 10, ip_dict)
    client1 = DataClient(1, channel1, 32, 128, 16, triplets_id=2)
    channel2 = Peer(2, "[::]:19003", 10, ip_dict)
    triplets_provider = TripletsProvider(2, channel2)

    matrix_multiply_2pc(client0, client1, triplets_provider)

    print("Matrix A hold by client 1:")
    print(client0.batch_data)
    print("Matrix B hold by client 2:")
    print(client1.other_paras[0])
    print("Matrix AB")
    print(client0.batch_data @ client1.other_paras[0])
    print("Matrix AB calculated by secret sharing")
    print(client0.shared_out_AB[1] + client1.shared_out_BA[0])

test_mm_2pc()