from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np


def generate_images(generator):
    noise = np.random.normal(0, 1, (1, 100))
    generated_images = generator.predict(noise)
    plt.imshow(generated_images[0, :, :, :])
    plt.axis("off")
    plt.savefig("generated.jpg")
    plt.show()
gan = load_model('generator_model.h5')
generate_images(gan)
