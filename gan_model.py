import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Input, Concatenate
from tensorflow.keras.models import Model
import numpy as np

def build_generator(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder
    x = Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)

    # Decoder
    x = Conv2DTranspose(128, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(64, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)

    outputs = Conv2D(3, (7, 7), activation='tanh', padding='same')(x)

    return Model(inputs, outputs, name='Generator')

def build_discriminator(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)

    outputs = Conv2D(1, (4, 4), padding='same')(x)

    return Model(inputs, outputs, name='Discriminator')

def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(256, 256, 1))  # Grayscale input
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    return Model(gan_input, gan_output, name='GAN')

def load_images(path_to_grayscale_images, path_to_color_images):
    """Load grayscale and color images from .npy files"""
    grayscale_images = np.load(path_to_grayscale_images)
    color_images = np.load(path_to_color_images)
    return grayscale_images, color_images

def initialize_models(input_shape_gray=(256, 256, 1), input_shape_color=(256, 256, 3)):
    """Initialize and compile the generator, discriminator and GAN models"""
    generator = build_generator(input_shape_gray)
    discriminator = build_discriminator(input_shape_color)
    gan = build_gan(generator, discriminator)

    # Compile models
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    
    return generator, discriminator, gan

def train_gan(generator, discriminator, gan, grayscale_images, color_images, 
              batch_size=32, num_epochs=100, save_interval=10):
    """Train the GAN model"""
    half_batch = batch_size // 2

    for epoch in range(num_epochs):
        for _ in range(len(grayscale_images) // batch_size):
            # Select a random half batch of real images
            idx = np.random.randint(0, grayscale_images.shape[0], half_batch)
            real_grayscale = grayscale_images[idx]
            real_color = color_images[idx]

            # Generate a half batch of new images
            generated_color = generator.predict(real_grayscale)

            # Train the discriminator
            d_loss_real = discriminator.train_on_batch(real_color, np.ones((half_batch, 1)))
            d_loss_fake = discriminator.train_on_batch(generated_color, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator (via the GAN model, where the discriminator is not trainable)
            g_loss = gan.train_on_batch(real_grayscale, np.ones((half_batch, 1)))

        # Print the progress
        print(f"{epoch + 1}/{num_epochs}, Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss}")

        # Save model checkpoints
        if (epoch + 1) % save_interval == 0:
            generator.save(f"generator_epoch_{epoch + 1}.h5")
            discriminator.save(f"discriminator_epoch_{epoch + 1}.h5")

def main():
    # Paths to image data
    path_to_grayscale_images = 'C:\\Users\\JamesHancock\\OneDrive\\03_Programming\\04_AI\\ImageColouriser\\TestImages\\Numpy\\Grayscale\\'
    path_to_color_images = 'C:\\Users\\JamesHancock\\OneDrive\\03_Programming\\04_AI\\ImageColouriser\\TestImages\\Numpy\\Colour\\'
    
    # Load images
    grayscale_images, color_images = load_images(path_to_grayscale_images, path_to_color_images)
    
    # Initialize models
    generator, discriminator, gan = initialize_models()
    
    # Train the GAN
    train_gan(generator, discriminator, gan, grayscale_images, color_images)

if __name__ == "__main__":
    main()


