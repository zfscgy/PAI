import threading
import time
from Communication.RPCComm import decode_ComputationData, encode_ComputationData, Peer
from Client.Client import BaseClient, MessageType, PackedMessage


def test_encode_decode():
    print("========== Test encoding & decoding =========")
    comp_msg = PackedMessage(header=MessageType.NULL, data=(12, 34))
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
    print("========= Test plain send =======")
    resp = channel0.send(1, PackedMessage(MessageType.NULL, "Hello from client 0"))
    print("Client0 get response:")
    print(resp)
    msg0 = channel1.receive(0, 1)
    print("Client1 received message:")
    print(msg0)
    print("========= Test send with key ======")
    resp = channel0.send(1, PackedMessage(MessageType.NULL, "Hello from client 0 with key 1", 1))
    print("Client 0 get response:")
    print(resp)
    msg01 = channel1.receive(0, 1, 1)
    print("Client1 received message:")
    print(msg01)
    print("========= Test send with key and multiple message ======")
    channel0.send(1, PackedMessage(MessageType.NULL, "Hello from client 0 with key 1", 1))
    channel0.send(1, PackedMessage(MessageType.NULL, "What's up from client 0 with key 1", 1))
    msg010 = channel1.receive(0, 1, 1)
    msg011 = channel1.receive(0, 1, 1)
    print("Client1 received message:")
    print(msg010)
    print(msg011)
    print("========= Test send with timeout ======")
    def recv():
        msg = channel1.receive(0, 3)
        print("Channel 1 received a delayed message")
        print(msg)
    def send():
        channel0.send(1, PackedMessage(MessageType.NULL, "A delayed message from client 0"))
    recv_th = threading.Thread(target=recv)
    recv_th.start()
    print("Start receive... wait for 1 sec to send")
    time.sleep(1)
    send_th = threading.Thread(target=send)
    send_th.start()
    recv_th.join()
    send_th.join()

    print("========== Test finished ========")


if __name__ == "__main__":
    test_encode_decode()
    test_rpc()
