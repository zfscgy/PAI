import numpy as np
from Client import Channel, DataClient, TripletsProvider
from Protocols import matrix_multiply_2pc

def test_mm_2pc():
    channel = Channel(3)
    client1 = DataClient(1, channel, 32, 64, 16)
    client2 = DataClient(2, channel, 32, 128, 16)
    triplets_provider = TripletsProvider(0, channel)

    matrix_multiply_2pc(client1, client2, triplets_provider)

    print("Matrix A hold by client 1:")
    print(client1.batch_data)
    print("Matrix B hold by client 2:")
    print(client2.other_paras[1])
    print("Matrix AB")
    print(client1.batch_data @ client2.other_paras[1])
    print("Matrix AB calculated by secret sharing")
    print(client1.shared_out_AB[2] + client2.shared_out_BA[1])

test_mm_2pc()