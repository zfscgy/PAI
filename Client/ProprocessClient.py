import threading
import pandas as pd
import numpy as np

from Client.Client import BaseClient
from Communication.Channel import BaseChannel
from Communication.Message import ComputationMessage, MessageType
from Utils.Log import Logger

from Crypto.Cipher import AES


class PreprocessClient(BaseClient):
    aes_key_len = 16

    def __init__(self, channel: BaseChannel, filepath: str,
                 align_server: int, other_data_clients: list, logger: Logger = None):
        super(PreprocessClient, self).__init__(channel, logger)
        self.filepath = filepath
        self.data = None
        self.sample_ids = None

        self.align_server = align_server
        self.other_clients = other_data_clients

        self.client_aes_keys = []
        self.client_aes_ivs = []
        self.shared_aes_key = None
        self.shared_aes_iv = None

        self.error = False

    def __padding(self, s):
        while len(s) % 16 != 0:
            s += '\0'
        return s

    def generate_raw_key_and_iv(self):
        """
        Generate AES key and iv
        """
        self.shared_aes_key = bytes(np.random.randint(0, 256, self.aes_key_len).tolist())
        self.shared_aes_iv = bytes(np.random.randint(0, 256, 16).tolist())

    def __send_aes_key_to(self, client_id):
        try:
            self.send_check_msg(client_id, ComputationMessage(MessageType.ALIGN_AES_KEY,
                                                              (self.shared_aes_key, self.shared_aes_iv)))
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
            msg = self.receive_check_msg(client_id, MessageType.ALIGN_AES_KEY)
            if len(msg.data[0]) != self.aes_key_len or len(msg.data[1]) != self.aes_key_len:
                self.logger.logE("Error, the received key ot iv from client %d is not of length %d but is of length %d" %
                                 (client_id, self.aes_key_len, len(msg.data)))
                self.error = True
            self.client_aes_keys.append(msg.data[0])
            self.client_aes_ivs.append(msg.data[1])
        except:
            self.logger.logE("Error while receiving aes key from client %d" % client_id)
            self.error = True

    def __receive_aes_keys(self):
        receiving_threads = []
        for client_id in self.other_clients:
            receiving_threads.append(threading.Thread(target=self.__receive_aes_key_from, args=(client_id,)))
            receiving_threads[-1].start()
        for receiving_thread in receiving_threads:
            receiving_thread.join()

    def __generate_shared_key(self):
        def nor_bytes(b1: bytes, b2: bytes):
            return bytes([x ^ y for (x, y) in zip(b1, b2)])
        for key in self.client_aes_keys:
            self.shared_aes_key = nor_bytes(self.shared_aes_key, key)
        for iv in self.client_aes_ivs:
            self.shared_aes_iv = nor_bytes(self.shared_aes_iv, iv)

    def __load_and_enc_data(self):
        try:
            self.data = pd.read_csv(self.filepath, header=None)
        except Exception as e:
            self.logger.logE("Error while read csv")
            raise e
        self.data[0] = self.data[0].astype(str)
        self.data = self.data.set_index(0)
        self.sample_ids = self.data.index.values.tolist()
        encrypted_ids = list()
        for sample_id in self.sample_ids:
            encrypted_ids.append(AES.new(self.shared_aes_key, AES.MODE_CBC, self.shared_aes_iv).
                                 encrypt(self.__padding(sample_id)))
        self.sample_ids = encrypted_ids

    def start_align(self):
        self.generate_raw_key_and_iv()
        self.__send_aes_keys()
        if self.error:
            self.logger.logE("Send aes keys failed, stop align")
            return False

        self.__receive_aes_keys()
        if self.error:
            self.logger.logE("Receive aes keys failed, stop align")
            return False

        self.__generate_shared_key()
        try:
            self.__load_and_enc_data()
        except:
            self.logger.logE("Error while loading and encrypting data. Stop align")
            return False

        try:
            self.send_check_msg(self.align_server, ComputationMessage(MessageType.ALIGN_ENC_IDS, self.sample_ids))
        except:
            self.logger.logE("Error while sending aligned ids to align server. Stop align")
            return False

        try:
            msg = self.receive_check_msg(self.align_server, MessageType.ALIGN_FINAL_IDS)
        except:
            self.logger.logE("Error while receiving ids intersection from align_server")
            return False

        encrypted_ids = msg.data
        selected_ids = list()
        for sample_id in encrypted_ids:
            selected_ids.append(AES.new(self.shared_aes_key, AES.MODE_CBC, self.shared_aes_iv).
                                    decrypt(sample_id).decode('utf8').replace('\0', ''))

        aligned_data = self.data.loc[selected_ids]
        aligned_data.to_csv(self.filepath[:-4] + "_aligned.csv", header=None, index=True)
        aligned_data.to_csv(self.filepath[:-4] + "_aligned_noindex.csv", header=None, index=False)


class MainPreprocessor(BaseClient):
    def __init__(self, channel: BaseChannel, data_clients: list, logger: Logger=None):
        super(MainPreprocessor, self).__init__(channel, logger)
        self.data_clients = data_clients
        self.client_id_lists = []
        self.aligned_ids = None

        self.error = False

    def __receive_encrypted_ids_from(self, client_id):
        try:
            msg = self.receive_check_msg(client_id, MessageType.ALIGN_ENC_IDS)
            self.client_id_lists.append(msg.data)
        except:
            self.logger.logE("Receiving encrypted ids from client %d failed")
            self.error = True

    def __receive_encrypted_ids(self):
        receiving_threads = []
        for client_id in self.data_clients:
            receiving_threads.append(threading.Thread(target=self.__receive_encrypted_ids_from, args=(client_id,)))
            receiving_threads[-1].start()
        for receiving_thread in receiving_threads:
            receiving_thread.join()

    def __send_aligned_ids_to(self, client_id):
        try:
            self.send_check_msg(client_id, ComputationMessage(MessageType.ALIGN_FINAL_IDS, self.aligned_ids))
        except:
            self.logger.logE("Sending aligned ids to client %d failed" % client_id)
            self.error = True

    def __send_aligned_ids(self):
        sending_threads = []
        for client_id in self.data_clients:
            sending_threads.append(threading.Thread(target=self.__send_aligned_ids_to, args=(client_id,)))
            sending_threads[-1].start()
        for sending_thread in sending_threads:
            sending_thread.join()

    def start_align(self):
        self.logger.log("ID Alignment start")

        self.__receive_encrypted_ids()
        if self.error:
            self.logger.logE("Error while receiving encrypted ids. Align stop")
            return False
        aligned_ids = set(self.client_id_lists[0])
        for id in self.client_id_lists[1:]:
            aligned_ids = aligned_ids & set(id)
        self.aligned_ids = list(aligned_ids)

        self.__send_aligned_ids()
        if self.error:
            self.logger.logE("Error while sending aligned ids. Align stop")
            return False

        self.logger.log("ID Alignment finished")
        return True