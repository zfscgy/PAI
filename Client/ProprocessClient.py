from Client.Client import BaseClient
from Communication.Channel import BaseChannel
from Communication.Message import ComputationMessage, MessageType
from Utils.Log import Logger

from Crypto import Random
from Crypto.Cipher import AES

import threading
import pandas as pd

class PreprocessClient(BaseClient):
    def __init__(self, channel: BaseChannel, filepath, prim_key: int, iv, key,
                 align_id: int, other_data_clients: list, logger: Logger = None):
        super(PreprocessClient, self).__init__(channel, logger)
        self.filepath = filepath
        self.data = None
        self.id = None
        self.prim_key = prim_key
        self.iv = iv
        self.align_id = align_id
        self.other_clients = other_data_clients
        self.aes_key = key
        self.aes_key_list = list()
        self.random_generator = Random.new().read

        self.error = False

    def __padding(self, s):
        while len(s) % 16 != 0:
            s += '\0'
        return s

    def __send_aes_key_to(self, client_id):
        try:
            self.send_check_msg(client_id, ComputationMessage(MessageType.AES_KEY, self.aes_key))
        except:
            self.logger.logE("Send raw aes key to client %d failed." % client_id)
            self.error = True

    def __send_aes_keys(self):
        sending_threads = []
        for client_id in self.other_clients:
            sending_threads.append(threading.Thread(target=self.__send_aes_key_to, args=(client_id,)))
            sending_threads[-1].start()
        for sending_thread in sending_threads:
            sending_thread.join()

    def __receive_aes_key_from(self, client_id):
        try:
            msg = self.receive_check_msg(client_id, MessageType.AES_KEY)
            self.aes_key_list.append(msg.data)
        except:
            self.logger.logE("Error while receiving aes key from client %d" % client_id)

    def __receive_aes_keys(self):
        receiving_threads = []
        for client_id in self.other_clients:
            receiving_threads.append(threading.Thread(target=self.__receive_aes_key_from, args=(client_id,)))
            receiving_threads[-1].start()
        for receiving_thread in receiving_threads:
            receiving_thread.join()

    def __generate_aes_key(self):
        self.__send_aes_keys()
        self.__receive_aes_keys()
        for key in self.aes_key_list:
            if len(self.aes_key) > len(key):
                self.aes_key = "".join([chr(ord(x) ^ ord(y)) for (x, y) in zip(self.aes_key[:len(key)], key)])
            else:
                self.aes_key = "".join([chr(ord(x) ^ ord(y)) for (x, y) in zip(self.aes_key, key[:len(self.aes_key)])])

    def __load_and_enc_data(self):
        self.data = pd.read_csv(self.filepath, header=None)
        self.data[self.prim_key] = self.data[self.prim_key].astype(str)
        self.id = self.data[self.prim_key].values.tolist()
        encrypted_ids = list()
        for id in self.id:
            cipher = AES.new(self.aes_key, AES.MODE_CBC, self.iv)
            encrypted_ids.append(cipher.encrypt(self.__padding(id)))
        self.id = encrypted_ids

    def start_align(self):
        self.__generate_aes_key()
        self.__load_and_enc_data()
        res = self.send_check_msg(self.align_id, ComputationMessage(MessageType.ALIGN_SEND, self.id))
        msg = self.receive_check_msg(self.align_id, MessageType.ALIGN_REC)
        aligned_id = msg.data
        dec_id = list()
        for id in aligned_id:
            cipher = AES.new(self.aes_key, AES.MODE_CBC,self.iv)
            dec_id.append(cipher.decrypt(id).decode('utf-8').replace('\0',''))
        aligned_id = dec_id
        aligned_data = self.data[self.data[self.prim_key].isin(aligned_id)]
        aligned_data.to_csv(self.filepath, header=None, index=None)


class MainProprocessor(BaseClient):
    def __init__(self, channel: BaseChannel, data_clients: list, logger: Logger=None):
        super(MainProprocessor, self).__init__(channel, logger)
        self.data_clients = data_clients
        self.id_lists = list()
        self.aligned_id = None

    def __receive_crypto_ids(self):
        for client in self.data_clients:
            msg = self.receive_check_msg(client, MessageType.ALIGN_SEND)
            self.id_lists.append(msg.data)

    def __send_aligned_id(self):
        for client in self.data_clients:
            res = self.send_check_msg(client, ComputationMessage(MessageType.ALIGN_REC, self.aligned_id))

    def start_align(self):
        self.__receive_crypto_ids()
        ald_id = set(self.id_lists[0])
        for id in self.id_lists[1:]:
            ald_id = ald_id & set(id)
        self.aligned_id = list(ald_id)
        self.__send_aligned_id()