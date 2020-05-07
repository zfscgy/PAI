import numpy as np
import threading
from Communication.RPCComm import ComputationRPCClient, ComputationRPCServer, \
    decode_ComputationData, encode_ComputationData, Peer
from Client import BaseClient, MessageType, ComputationMessage


def test_encode_decode():
    print("========== Test encoding & decoding =========")
    comp_msg = ComputationMessage(header=MessageType.DATA_DIM, data=(12, 34))
    encoded_msg = encode_ComputationData(comp_msg)
    decoded_msg = decode_ComputationData(encoded_msg)
    print("Raw:", comp_msg, sep="\n")
    print("Encoded:", encoded_msg, sep="\n")
    print("Decoded:", decoded_msg, sep="\n")
    print("========= Test finished ========")


def test_rpc():
    print("========== Test rpc clients ========")
    ip_dict = {
        0: "127.0.0.1:19001",
        1: "127.0.0.1:19002"
    }
    channel0 = Peer(0, "[::]:19001", 10, ip_dict)
    channel1 = Peer(1, "[::]:19002", 10, ip_dict)
    client0 = BaseClient(0, channel0)
    client1 = BaseClient(1, channel1)
    def client0_send():
        resp = client0.send_msg(1, ComputationMessage(MessageType.NULL, "Hello from client 0"))
        print("Client0 get response:")
        print(resp)
    def client1_recv():
        msg0 = client1.receive_msg(0)
        print("Client1 get response:")
        print(msg0)
    client0thread = threading.Thread(target=client0_send, name="Client0:Send")
    client1thread = threading.Thread(target=client1_recv, name="Client1:Receive")
    client0thread.start()
    client1thread.start()
    client0thread.join()
    client1thread.join()
    print("========== Test finished ========")

if __name__ == "__main__":
    test_encode_decode()
    test_rpc()
