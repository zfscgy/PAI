import tensorflow as tf
from Communication.Channel import BaseChannel
from Client.Client import BaseClient
k = tf.keras

class MainTFClient(BaseClient):
    def __init__(self, client_id, channel: BaseChannel, logger: Logger=None):
        super(MainTFClient, self).__init__(client_id, channel, logger)
