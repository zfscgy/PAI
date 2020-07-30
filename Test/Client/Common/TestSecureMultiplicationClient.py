import threading
import time
import numpy as np
from Client.MPCClient import MPCClientParas
from Client.Common.SecureMultiplicationClient import SecureMultiplicationClient
from Client.MPCProviders.TripletProducer import TripletProducer
from Communication.RPCComm import Peer
from Utils.Log import Logger


print("==========Test Secure Multiplication Client==============")
ip_dict = {
    0: '127.0.0.1:8900',
    1: '127.0.0.1:8901',
    2: '127.0.0.1:8902'
}
channel0 = Peer(0, '0.0.0.0:8900', 3, ip_dict, logger=Logger(prefix="0:"))
channel1 = Peer(1, '0.0.0.0:8901', 3, ip_dict, logger=Logger(prefix="1:"))
channel2 = Peer(2, '0.0.0.0:8902', 3, ip_dict, logger=Logger(prefix="triplet:"))
mul_client_0 = SecureMultiplicationClient(channel0, logger=Logger(prefix="0:"))
mul_client_1 = SecureMultiplicationClient(channel1, logger=Logger(prefix="1:"))
triplet_client = TripletProducer(channel2, Logger(prefix="triplet:"), MPCClientParas([0, 1], -1, -1, 2), [0, 1])

threading.Thread(target=triplet_client.start_listening).start()
time.sleep(3)
mat0 = np.random.uniform(0, 1, [5, 10])
mat1 = np.random.uniform(0, 1, [10, 20])
mul_0_th = threading.Thread(target=mul_client_0.multiply_AB_with, args=(1, 2, (10, 20), mat0))
mul_1_th = threading.Thread(target=mul_client_1.multiply_BA_with, args=(0, 2, (5, 10), mat1))
mul_0_th.start()
mul_1_th.start()
mul_0_th.join()
mul_1_th.join()
out0 = mul_client_0.product
out1 = mul_client_1.product
print("Expected: ", mat0 @ mat1)
print("Using MPC: ", out0 + out1)
print("==============Test finished====================")