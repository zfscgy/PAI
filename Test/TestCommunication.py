import numpy as np
from Communication.RPCComm import ComputationRPCClient, ComputationRPCServer, \
    decode_ComputationData, encode_ComputationData
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
    pass


if __name__ == "__main__":
    test_encode_decode()
