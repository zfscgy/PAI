import threading
import pandas as pd
import numpy as np
import pathlib
from Client.Client import BaseClient
from Communication.Channel import BaseChannel
from Communication.Message import PackedMessage, MessageType
from Utils.Log import Logger
from Client.MPCClient import MPCClient, MPCClientParas, ClientMode

from Crypto.Cipher import AES


class PreprocessClient(MPCClient):
    aes_key_len = 16

    def __init__(self, channel: BaseChannel, logger: Logger, mpc_paras: MPCClientParas,
                 filepath: str, out_dir: str, cols: list = None):
        super(PreprocessClient, self).__init__(channel, logger, mpc_paras)
        self.filepath = filepath
        self.out_dir = out_dir
        self.cols = cols

        self.other_data_client_ids = self.feature_client_ids + [self.label_client_id]
        self.other_data_client_ids.remove(self.client_id)

        self.data = None
        self.sample_ids = None

        self.client_aes_keys = []
        self.client_aes_ivs = []
        self.shared_aes_key = None
        self.shared_aes_iv = None

        self.error = False

        self.out_indexed_file = None
        self.train_data_path = ""
        self.test_data_path = ""

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
            self.send_check_msg(client_id, PackedMessage(MessageType.ALIGN_AES_KEY,
                                                         (self.shared_aes_key, self.shared_aes_iv)))
        except:
            self.logger.logE("Send raw aes key to client %d failed." % client_id)
            self.error = True

    def __send_aes_keys(self):
        sending_threads = []
        for client_id in self.other_data_client_ids:
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
        for client_id in self.other_data_client_ids:
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
            self.data = pd.read_csv(self.filepath)
        except Exception as e:
            self.logger.logE("Error while read csv")
            raise e
        self.data.iloc[:, 0] = self.data.iloc[:, 0].astype(str)
        self.data = self.data.set_index(self.data.columns[0])
        self.sample_ids = self.data.index.values.tolist()
        encrypted_ids = list()
        for sample_id in self.sample_ids:
            encrypted_ids.append(AES.new(self.shared_aes_key, AES.MODE_CBC, self.shared_aes_iv).
                                 encrypt(self.__padding(sample_id)))
        self.sample_ids = encrypted_ids

    def start_align(self):
        self.logger.log("Start generating random keys and ivs")
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
        self.logger.log("Shared key generated. Start encrypt ids.")
        try:
            self.__load_and_enc_data()
        except:
            self.logger.logE("Error while loading and encrypting data. Stop align")
            return False
        self.logger.log("Start sending encrypted ids to main preprocessor")
        try:
            self.send_check_msg(self.main_client_id, PackedMessage(MessageType.ALIGN_ENC_IDS, self.sample_ids))
        except:
            self.logger.logE("Error while sending aligned ids to align server. Stop align")
            return False

        try:
            msg = self.receive_check_msg(self.main_client_id, MessageType.ALIGN_FINAL_IDS)
        except:
            self.logger.logE("Error while receiving ids intersection from align_server")
            return False

        self.logger.log("Received aligned ids. Start making aligned data.")
        encrypted_ids = msg.data
        selected_ids = list()
        for sample_id in encrypted_ids:
            selected_ids.append(AES.new(self.shared_aes_key, AES.MODE_CBC, self.shared_aes_iv).
                                decrypt(sample_id).decode('utf8').replace('\0', ''))

        aligned_data = self.data.loc[selected_ids]

        file_name = pathlib.Path(self.filepath).name

        self.out_indexed_file = self.out_dir + file_name[:-4] + "_aligned.csv"
        aligned_data.to_csv(self.out_indexed_file, index=True)
        test_size = int(len(selected_ids) / 5)
        train_data = aligned_data.iloc[:-test_size]
        test_data = aligned_data.iloc[-test_size:]

        self.train_data_path = self.out_dir + "train.csv"
        self.test_data_path = self.out_dir + "test.csv"
        if self.cols is None:
            self.cols = self.data.columns
        if isinstance(self.cols[0], int):
            self.cols = self.data.columns[self.cols]
        if not set(self.cols) <= set(train_data.columns):
            self.logger.logE("Selected columns not in the csv.")
            return False
        train_data[self.cols].to_csv(self.train_data_path, header=False, index=False)
        test_data[self.cols].to_csv(self.test_data_path, header=False, index=False)

        self.logger.log("Align finished, aligned file saved in " + self.out_dir)
        return True


class MainPreprocessor(MPCClient):
    def __init__(self, channel: BaseChannel, logger: Logger, mpc_paras: MPCClientParas):
        super(MainPreprocessor, self).__init__(channel, logger, mpc_paras)
        self.data_clients = self.feature_client_ids + [self.label_client_id]
        self.client_dataid_lists = []
        self.aligned_ids = None

        self.error = False

    def __receive_encrypted_ids_from(self, client_id):
        try:
            msg = self.receive_check_msg(client_id, MessageType.ALIGN_ENC_IDS)
            self.client_dataid_lists.append(msg.data)
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
            self.send_check_msg(client_id, PackedMessage(MessageType.ALIGN_FINAL_IDS, self.aligned_ids))
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
        aligned_ids = set(self.client_dataid_lists[0])
        for id in self.client_dataid_lists[1:]:
            aligned_ids = aligned_ids & set(id)
        self.aligned_ids = list(aligned_ids)

        self.__send_aligned_ids()
        if self.error:
            self.logger.logE("Error while sending aligned ids. Align stop")
            return False

        self.logger.log("ID Alignment finished")
        return True
