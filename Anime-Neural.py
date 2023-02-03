import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

generated='generated'

# Generator model
def make_generator_model():
    print("Generator model in progress")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128 * 7 * 7, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Reshape((7, 7, 128)))
    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

# Discriminator model
def make_discriminator_model():
    print("Discriminator model in progress")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape((28 * 28 * 3,), input_shape=(28, 28, 3)))
    model.add(tf.keras.layers.Dense(512, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    discriminator = tf.keras.Sequential()

    return model


def resume_training(generator, discriminator, gan, model_dir, animecutepics):
    print("Resuming training from checkpoint:", model_dir)
    generator.load_weights(os.path.join(model_dir, 'generator_weights.h5'))
    discriminator.load_weights(os.path.join(model_dir, 'discriminator_weights.h5'))
    gan.load_weights(os.path.join(model_dir, 'gan_weights.h5'))
    discriminator.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    gan.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy')
    training_loop(generator, discriminator, gan, model_dir, animecutepics)

def training_loop(generator, discriminator, gan, model_dir, animecutepics):
    # Train the GAN
    print("Training...")
    num_epochs = 10000
    batch_size = 64
    for epoch in tqdm(range(num_epochs)):
        idx = np.random.randint(0, animecutepics.shape[0], batch_size)
        real_images = animecutepics[idx]
        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_images = generator.predict(noise)
        x = np.concatenate([real_images, fake_images])
        y = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
        d_loss, d_acc = discriminator.train_on_batch(x, y)

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        y = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, y)

        # losses
        if (epoch + 1) % 10 == 0:
            print("Epoch:", epoch + 1, "Discriminator Loss:", d_loss, "Accuracy:", d_acc, "Generator Loss:", g_loss)
            plt.imshow(fake_images[0, :, :, :])
            plt.savefig(os.path.join(generated, str(epoch)+".png"))
            #plt.show()
        if (epoch + 1) % 50 == 0:
            # Save weights
            generator.save_weights(os.path.join(model_dir, 'generator_weights.h5'))
            discriminator.save_weights(os.path.join(model_dir, 'discriminator_weights.h5'))
            gan.save_weights(os.path.join(model_dir, 'gan_weights.h5'))

def main():
    
    np.random.seed(42)
    tf.random.set_seed(42)
    model_dir = 'models'
    # Load animu UWU images
    animecutepics = []
    animecutepics_path = 'data/cropped28'
    for filename in os.listdir(animecutepics_path):
        if filename.endswith(".jpg"):
            animecutepics.append(plt.imread(os.path.join(animecutepics_path, filename)))
            print("Loading the pic "+ filename)
    animecutepics = np.array(animecutepics)

    # Normalize images
    animecutepics = (animecutepics - 127.5) / 127.5
    # Generate the models
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    # Compile the models
    g_optimizer = tf.keras.optimizers.Adam(1e-4)
    d_optimizer = tf
    # Create the generator and discriminator models

    print("Discriminator model DONE")
    generator = make_generator_model()
    print("Discriminator model DONE")
    discriminator = make_discriminator_model()

    # Compile the discriminator model
    print("Discriminator model compiling in progress...")
    discriminator.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # Combine the generator and discriminator into a GAN
    print("Combining.. GAN + Discriminator")
    gan = tf.keras.Sequential([generator, discriminator])
    print("Compiling in progress...")
    gan.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        training_loop(generator, discriminator, gan, model_dir , animecutepics)
    else:
        resume_training(generator, discriminator, gan, model_dir, animecutepics)

main()