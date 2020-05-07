from Client import DataClient, TripletsProvider


def matrix_multiply_2pc(client_0: DataClient, client_1: DataClient, triplet_provider: TripletsProvider):
    c0 = client_0.client_id
    c1 = client_1.client_id
    triplet_provider.start_listening()
    thread0 = client_0.start_calc_first_layer(c1)
    thread1 = client_1.start_calc_first_layer(c0)
    thread0.join()
    thread1.join()
    triplet_provider.stop_listening()