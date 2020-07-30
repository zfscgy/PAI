import threading
import numpy as np
import traceback
from Communication.Message import PackedMessage, MessageType
from Communication.Channel import BaseChannel
import Client.MPCClient as MPCC
from Client.Common.BroadcastClient import BroadcastClient
from Utils.Log import Logger
import tensorflow as tf
k = tf.keras


class MainClient(MPCC.MainClient):
    def __init__(self, channel: BaseChannel, logger: Logger,
                 mpc_params: MPCC.MPCClientParas,
                 in_dim, out_dim, layers, batch_size, test_batch_size, test_per_batches, learning_rate, max_iter):
        super(MainClient, self).__init__(channel, logger, mpc_params)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = layers
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.test_per_batches = test_per_batches
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        self.error = False
        self.finished = False
        #
        self.broadcaster = BroadcastClient(self.channel, self.logger)

        self.data_client_outs = dict()
        #
        self.input_tensor = None
        self.network = None
        self.optimizer = None
        self.network_out = None
        self.gradient_tape = None

        self.test_per_batch = None

        self.n_rounds = 0

    def _build_keras_model(self, network: k.Model, optimizer: k.optimizers, in_dim):
        self.network = network
        self.optimizer = optimizer
        self.network.build((None, in_dim))
        # Do a prediction to initialize the network
        o = self.network(np.random.normal(size=[100, in_dim]))

    def _build_mlp_model(self, input_dim: int, layers: list):
        network = k.Sequential([k.layers.Activation(k.activations.sigmoid, input_shape=(input_dim,))] +
                               [k.layers.Dense(layer, activation=k.activations.sigmoid) for layer in layers])
        optimizer = k.optimizers.SGD()
        self._build_keras_model(network, optimizer, input_dim)

    def _before_training(self):
        try:
            self._build_mlp_model(self.in_dim, self.layers)
        except:
            self.logger.logE("Build tensorflow model failed. Stop training.")
            return False

        self.client_dims = self.broadcaster.receive_all(self.feature_client_ids, MessageType.SharedNN_ClientDim)
        if self.broadcaster.error:
            self.logger.logE("Gather clients' dims failed. Stop training.")
            return False
        train_config = {
            "client_dims": self.client_dims,
            "out_dim": self.in_dim,
            "batch_size": self.batch_size,
            "test_batch_size": self.test_batch_size,
            "learning_rate": self.learning_rate
        }
        self.broadcaster.broadcast(self.feature_client_ids + [self.label_client_id], PackedMessage(
            MessageType.SharedNN_TrainConfig, train_config
        ))
        if self.broadcaster.error:
            self.logger.logE("Broadcast training config message failed. Stop training.")
            return False
        return True

    def _forward(self):
        client_outs = self.broadcaster.receive_all(self.feature_client_ids, MessageType.SharedNN_FeatureClientOut)
        if self.broadcaster.error:
            self.logger.logE("Gather clients' outputs failed. Stop training")
            return False

        output_parts = []
        for data_client in self.feature_client_ids:
            output_part = client_outs[data_client][0]
            for other_client in self.feature_client_ids:
                if other_client != data_client:
                    output_part += client_outs[data_client][1][other_client] + \
                                   client_outs[other_client][2][data_client]
            output_parts.append(output_part)
        first_layer = sum(output_parts)

        self.input_tensor = tf.Variable(first_layer, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            self.network_out = self.network(self.input_tensor)
        self.gradient_tape = tape
        return True

    def _compute_loss(self):
        try:
            mode_str = "Train"
            if self.mpc_mode is MPCC.ClientMode.Test:
                mode_str = "Test"
            if self.n_rounds == self.max_iter:
                self.finished = True
                mode_str += "-Stop"
            self.send_check_msg(self.label_client_id, PackedMessage(MessageType.SharedNN_MainClientOut, (self.network_out.numpy(), mode_str)))
            self.grad_on_output = self.receive_check_msg(self.label_client_id, MessageType.SharedNN_MainClientGradLoss).data[0]
        except:
            self.logger.logE("Get gradients from label client failed. Stop training.")
            return False
        return True

    def _backward(self):
        if self.mpc_mode is MPCC.ClientMode.Train:
            grad_on_output = self.grad_on_output
            if len(self.network.trainable_variables) != 0:
                model_jacobians = self.gradient_tape.jacobian(self.network_out, self.network.trainable_variables)
                model_grad = [tf.reduce_sum(model_jacobian * (tf.reshape(grad_on_output.astype(np.float32),
                                                                         list(grad_on_output.shape) + [1 for _ in range(len(model_jacobian.shape) - 2)]) +
                                                              tf.zeros_like(model_jacobian, dtype=model_jacobian.dtype)),
                                            axis=[0, 1]) for model_jacobian in model_jacobians]
                self.optimizer.apply_gradients(zip(model_grad, self.network.trainable_variables))
            input_jacobian = self.gradient_tape.jacobian(self.network_out, self.input_tensor)
            input_grad = tf.reduce_sum(input_jacobian * (tf.reshape(grad_on_output.astype(np.float32),
                                                                    list(grad_on_output.shape) + [1 for i in range(len(input_jacobian.shape) - 2)]) +
                                       tf.zeros_like(self.input_tensor, dtype=self.input_tensor.dtype)),
                                       axis=[0, 1]).numpy()
        else:
            input_grad = None

        if self.n_rounds == self.max_iter:
            self.broadcaster.broadcast(self.feature_client_ids,
                                       PackedMessage(MessageType.SharedNN_FeatureClientGrad, (input_grad, "Stop")))
            try:
                self.send_check_msg(self.crypto_producer_id, PackedMessage(MessageType.Common_Stop, None, key="Stop"))
            except:
                self.logger.logW("Send stop message to triplet provider failed.")

        elif (self.n_rounds + 1) % self.test_per_batches == 0:
            self.mpc_mode = MPCC.ClientMode.Test
            self.broadcaster.broadcast(self.feature_client_ids,
                                       PackedMessage(MessageType.SharedNN_FeatureClientGrad, (input_grad, "Continue-Test")))
        else:
            self.mpc_mode = MPCC.ClientMode.Train
            self.broadcaster.broadcast(self.feature_client_ids,
                                       PackedMessage(MessageType.SharedNN_FeatureClientGrad, (input_grad, "Continue-Train")))

        if self.broadcaster.error:
            self.logger.logE("Broadcast feature client's grads failed. Stop training.")
            return False

        return True

    def start_train(self) -> bool:
        self._before_training()
        while True:
            if not self._forward():
                return False
            if not self._compute_loss():
                return False
            if not self._backward():
                return False
            if self.finished:
                return True
            self.n_rounds += 1
