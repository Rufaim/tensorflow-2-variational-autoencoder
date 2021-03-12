import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self,model: tf.Module,latent_dim,seed=None,dtype=tf.float32):
        super(Encoder, self).__init__(dtype=dtype)
        self.model = model
        self.latent_dim = latent_dim
        self.seed = seed
        self._random_generator = tf.random.Generator.from_seed(seed=seed)

    def build(self, input_shape):
        init = tf.keras.initializers.glorot_uniform(seed=self.seed)
        self.out_mean = tf.keras.layers.Dense(self.latent_dim,kernel_initializer=init)
        self.out_std = tf.keras.layers.Dense(self.latent_dim,kernel_initializer=init)

    @tf.function
    def call(self,input,training=False):
        out = self.model(input,training)
        mean = self.out_mean(out)
        logstd = self.out_std(out)
        return mean, logstd

    @tf.function
    def sample(self,input, training=False):
        mean, logstd = self(input,training)
        n = self._random_generator.normal(shape=mean.shape)
        return  mean + tf.exp(0.5 * logstd) * tf.stop_gradient(n), mean, logstd

