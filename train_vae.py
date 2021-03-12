import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pyplot
from models import Encoder, VariationalAutoEncoder



BATCH_SIZE = 32
MNIST_SHAPE = [28, 28, 1]
LATENT_DIM = 12
EPOCHS = 25
LEARNING_RATE = 1e-3
LOGDIR = "logs"
SEED = 42



def preprocess_images(images):
    images = images.reshape([images.shape[0]] + MNIST_SHAPE) / 255
    return images.astype(np.float32)

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(len(train_images),seed=SEED).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(len(test_images),seed=SEED).batch(BATCH_SIZE)

init = tf.keras.initializers.glorot_normal(seed=SEED)
encoder_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32,kernel_size=3,strides=(2,2),activation=tf.nn.relu, padding="same",kernel_initializer=init),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=(2,2),activation=tf.nn.relu,kernel_initializer=init),
    tf.keras.layers.Flatten(),
])
encoder = Encoder(encoder_model,LATENT_DIM,seed=SEED)
decoder_model = tf.keras.Sequential([
    tf.keras.layers.Dense(units= 3 * 3 * 32, activation=tf.nn.relu),
    tf.keras.layers.Reshape(target_shape=(3, 3, 32)),
    tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3,strides=(2,2),activation=tf.nn.relu,padding="valid",kernel_initializer=init),
    tf.keras.layers.UpSampling2D(size=(2,2)),
    tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=3,strides=(2,2),activation=tf.nn.relu,padding="same",kernel_initializer=init),
    tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1,1), padding='same',kernel_initializer=init)
])
vae = VariationalAutoEncoder(encoder,decoder_model,MNIST_SHAPE,LEARNING_RATE,seed=SEED)

### warmup
warmup_input = np.zeros([1]+MNIST_SHAPE,dtype=np.float32)
decoder_model(encoder.sample(warmup_input)[0])
###

if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

test_portion = next(iter(test_dataset.take(1)))[:16,...]
for epoch in range(EPOCHS):
    for imgs in train_dataset:
        vae.train_step(imgs)

    loss = tf.keras.metrics.Mean()
    for imgs in test_dataset:
        loss(vae.loss(imgs))
    print(f"Epoch {epoch} | Loss: {loss.result()}")

    prediction = vae.predict(test_portion)
    fig = pyplot.figure(figsize=(4,4))
    for i in range(prediction.shape[0]):
        pyplot.subplot(4,4,i+1)
        pyplot.imshow(prediction[i,...,0],cmap="gray")
        pyplot.axis("off")
    pyplot.savefig(os.path.join(LOGDIR,f"epoch_{epoch}.png"))
    pyplot.close(fig)

