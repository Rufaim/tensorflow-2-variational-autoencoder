import tensorflow as tf
import numpy as np
from .encoder import Encoder


class VariationalAutoEncoder(object):
    def __init__(self, encoder: Encoder, decoder: tf.Module, inputs_shape: list, learning_rate=1e-3, seed=None, dtype=np.float32):
        self.encoder = encoder
        self.decoder = decoder
        self.inputs_shape = inputs_shape
        self.dtype = dtype
        if type(dtype) is str:
            self.dtype = np.dtype(dtype).type
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self._random_generator = np.random.RandomState(seed=seed)

    def predict(self,inputs):
        samples, _, _ = self.encoder.sample(inputs, True)
        return tf.nn.sigmoid(self.decoder(samples)).numpy()

    def loss(self, inputs):
        samples, mean, logstd = self.encoder.sample(inputs, True)
        decoded_imgs = self.decoder(samples)
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=decoded_imgs, labels=inputs)
        entropy = tf.reduce_sum(entropy, axis=list(range(1, len(entropy.shape))))
        samples_normal_loss = 0.5*tf.reduce_sum(samples**2,axis=-1)
        kl_loss = 0.5 * tf.reduce_sum((samples - mean) ** 2 / tf.math.exp(logstd) + logstd, axis=-1, keepdims=True)
        elbo_loss = tf.reduce_mean(samples_normal_loss - kl_loss + entropy)
        mae_loss = tf.reduce_mean((inputs-tf.nn.sigmoid(decoded_imgs))**2)
        return elbo_loss + mae_loss

    def train_step(self,inputs):
        inputs_ =  np.asanyarray(inputs,dtype=self.dtype).reshape([-1] + self.inputs_shape)

        with tf.GradientTape() as tape:
            loss = self.loss(inputs_)

        vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, vars)
        self._optimizer.apply_gradients(zip(gradients, vars))
        return loss

