import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from csv import writer
import datetime

#Current date
now = datetime.datetime.now()
#Folder for generated fake pictures
generated='generated'

def gpu():
    # Check if an AMD GPU is available (i have AMD D: )
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0 and tf.test.is_gpu_available(cuda_only=False):
        # Use the AMD GPU
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        raise Exception("No AMD GPU found")
    # Verify the GPU is being used
    print("Using GPU:", tf.test.gpu_device_name())

# Generator model
def make_generator_model():
    print("Generator model in progress")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128 * 16 * 16, activation="relu", input_shape=(100, )))
    model.add(tf.keras.layers.Reshape((16, 16, 128)))
    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(128, kernel_size=3, padding="same"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, padding="same"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Conv2D(3, kernel_size=3, padding="same"))
    model.add(tf.keras.layers.Activation("tanh"))
    return model

# Discriminator model
def make_discriminator_model():
    print("Discriminator model in progress")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape((64 * 64 * 3,), input_shape=(64, 64, 3)))
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def define_gan(discriminator, generator):
    discriminator.trainable = False
    model = tf.keras.Sequential([generator, discriminator])
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

#Resume loop
def resume_training(generator, discriminator, gan, model_dir, animecutepics):
    print("Resuming training from checkpoint:", model_dir)
    generator.load_weights(os.path.join(model_dir, 'generator_weights.h5'))
    discriminator.load_weights(os.path.join(model_dir, 'discriminator_weights.h5'))
    gan.load_weights(os.path.join(model_dir, 'gan_weights.h5'))
    training_loop(generator, discriminator, gan, model_dir, animecutepics)

def training_loop(generator, discriminator, gan, model_dir, animecutepics):
    # Train the GAN
    print("Training...")
    num_epochs = 10000
    batch_size = 512
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
        if (epoch + 1) % 10 == 0:
            print("Epoch:", epoch + 1, "Discriminator Loss:", d_loss, "Accuracy:", d_acc, "Generator Loss:", g_loss)
            plt.imshow(fake_images[0, :, :, :])
            plt.savefig(os.path.join(generated, str(epoch)+now.strftime("%m%d%Y%H%M%S")+".png"))
            data_csv(epoch,d_loss,d_acc,g_loss)
            #plt.show()
        if (epoch + 1) % 50 == 0:
            # Save weights
            generator.save_weights(os.path.join(model_dir, 'generator_weights.h5'))
            discriminator.save_weights(os.path.join(model_dir, 'discriminator_weights.h5'))
            gan.save_weights(os.path.join(model_dir, 'gan_weights.h5'))
def main():
    np.random.seed(42)
    tf.set_random_seed(42)
    model_dir = 'models'
    # Load animu UWU images
    animecutepics = []
    animecutepics_path = 'data/64x64'
    for filename in os.listdir(animecutepics_path):
        if filename.endswith(".jpg"):
            image = plt.imread(os.path.join(animecutepics_path, filename))
            #grayscale_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
            animecutepics.append(image)
            print("Loading the pic "+ filename)
    animecutepics = np.array(animecutepics)
    # Normalize images
    animecutepics = (animecutepics - 127.5) / 127.5
    print("Generator compile DONE")
    generator = make_generator_model()
    print("Discriminator compile DONE")
    discriminator = make_discriminator_model()
    print("Discriminator model compiling in progress...")
    print("Combining.. GAN + Discriminator")
    gan = define_gan(discriminator,generator)
    print("Compiling in progress...")
    # training_loop(generator, discriminator, gan, model_dir , animecutepics)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        training_loop(generator, discriminator, gan, model_dir , animecutepics)
    else:
        resume_training(generator, discriminator, gan, model_dir, animecutepics)
        
#CSV Telemetry 
def data_csv(epoch, d_loss, d_acc, g_loss):
    List = [now.strftime("%m/%d/%Y, %H:%M:%S") , epoch, d_loss, d_acc, g_loss]
    with open('epoch_info.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(List)
        f_object.close()
        
if __name__ == '__main__':
    main()