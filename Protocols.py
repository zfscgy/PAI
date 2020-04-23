from Client import DataClient, TripletsProvider


def matrix_multiply_2pc(client_0: DataClient, client_1: DataClient, triplet_provider: TripletsProvider):
    assert client_0.channel == client_1.channel and client_1.channel == triplet_provider.channel,\
        "All clients must share the same channel"
    c0 = client_0.client_id
    c1 = client_1.client_id
    tid = client_0.channel.triplets_id

    # Clients load data
    client_0.get_next_batch()
    client_1.get_next_batch()

    client_0.send_data_sim(c1)
    client_1.send_data_sim(c0)
    client_0.receive_msg(c1)
    client_1.receive_msg(c0)

    client_0.set_triplet_AB(c1)
    client_1.set_triplet_BA(c0)
    triplet_provider.receive_msg(c0)
    triplet_provider.receive_msg(c1)
    client_0.receive_msg(tid)
    client_1.receive_msg(tid)

    client_0.share_data(c1)
    client_1.share_para(c0)
    client_0.receive_msg(c1)
    client_1.receive_msg(c0)

    client_0.recover_own_value(c1)
    client_1.recover_own_value(c0)
    client_0.receive_msg(c1)
    client_1.receive_msg(c0)

    client_0.recover_other_value(c1)
    client_1.recover_other_value(c0)
    client_0.receive_msg(c1)
    client_1.receive_msg(c0)

    client_0.get_shared_out_AB(c1)
    client_1.get_shared_out_BA(c0)