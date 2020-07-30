import numpy as np
from Client.Client import BaseClient
from Communication.Channel import BaseChannel
from Communication.Message import MessageType, PackedMessage
from Utils.Log import Logger


class SecureMultiplicationClient(BaseClient):
    def __init__(self, channel: BaseChannel, logger: Logger):
        super(SecureMultiplicationClient, self).__init__(channel, logger)

    def multiply_AB_with(self, client_id: int, triplet_id: int, client_shape: tuple, matA: np.ndarray):
        try:
            self.send_check_msg(triplet_id,
                                PackedMessage(MessageType.Triplet_Set, (1, client_id, matA.shape, client_shape)))
            triplet_msg = self.receive_check_msg(triplet_id, MessageType.Triplet_Array, key=client_id).data
            shared_U_self, shared_V_self, shared_W_self = triplet_msg[1:]
        except:
            self.logger.logE("Get triplet arrays failed. Stop multiplication with client %d." % client_id)
            return False
        try:
            shared_A_self = matA * np.random.uniform(0, 1, matA.shape)
            shared_A_other = matA - shared_A_self
            self.send_check_msg(client_id, PackedMessage(MessageType.MUL_Mat_Share, shared_A_other))
            shared_B_self = self.receive_check_msg(client_id, MessageType.MUL_Mat_Share).data


        except:
            self.logger.logE("Swap mat shares with client. Stop multiplication with client %d." % client_id)
            return False
        try:
            shared_A_sub_U_self = shared_A_self - shared_U_self
            self.send_check_msg(client_id, PackedMessage(MessageType.MUL_AsubU_Share, shared_A_sub_U_self))
            shared_A_sub_U_other = self.receive_check_msg(client_id, MessageType.MUL_AsubU_Share).data

            shared_B_sub_V_self = shared_B_self - shared_V_self
            self.send_check_msg(client_id, PackedMessage(MessageType.MUL_BsubV_Share, shared_B_sub_V_self))
            shared_B_sub_V_other = self.receive_check_msg(client_id, MessageType.MUL_BsubV_Share).data
        except:
            self.logger.logE("Swap A-U and B-V with client failed. Stop multiplication with client %d" % client_id)
            return False

        A_sub_U = shared_A_sub_U_self + shared_A_sub_U_other
        B_sub_V = shared_B_sub_V_self + shared_B_sub_V_other
        self.product = A_sub_U @ B_sub_V + shared_U_self @ B_sub_V + A_sub_U @ shared_V_self + shared_W_self
        return True

    def multiply_BA_with(self, client_id: int, triplet_id: int, client_shape: tuple, matB: np.ndarray):
        try:
            self.send_check_msg(triplet_id,
                                PackedMessage(MessageType.Triplet_Set, (2, client_id, matB.shape, client_shape)))
            triplet_msg = self.receive_check_msg(triplet_id, MessageType.Triplet_Array, key=client_id).data
            shared_V_self, shared_U_self, shared_W_self = triplet_msg[1:]
        except:
            self.logger.logE("Get triplet arrays failed. Stop multiplication with client %d." % client_id)
            return False
        try:
            shared_B_self = matB * np.random.uniform(0, 1, matB.shape)
            shared_B_other = matB - shared_B_self
            self.send_check_msg(client_id, PackedMessage(MessageType.MUL_Mat_Share, shared_B_other))
            shared_A_self = self.receive_check_msg(client_id, MessageType.MUL_Mat_Share).data


        except:
            self.logger.logE("Swap mat shares with client. Stop multiplication with client %d." % client_id)
            return False
        try:
            shared_A_sub_U_self = shared_A_self - shared_U_self
            self.send_check_msg(client_id, PackedMessage(MessageType.MUL_AsubU_Share, shared_A_sub_U_self))
            shared_A_sub_U_other = self.receive_check_msg(client_id, MessageType.MUL_AsubU_Share).data

            shared_B_sub_V_self = shared_B_self - shared_V_self
            self.send_check_msg(client_id, PackedMessage(MessageType.MUL_BsubV_Share, shared_B_sub_V_self))
            shared_B_sub_V_other = self.receive_check_msg(client_id, MessageType.MUL_BsubV_Share).data
        except:
            self.logger.logE("Swap A-U and B-V with client failed. Stop multiplication with client %d" % client_id)
            return False

        A_sub_U = shared_A_sub_U_self + shared_A_sub_U_other
        B_sub_V = shared_B_sub_V_self + shared_B_sub_V_other
        self.product = shared_U_self @ B_sub_V + A_sub_U @ shared_V_self + shared_W_self
        return True
