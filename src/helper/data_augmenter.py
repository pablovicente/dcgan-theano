"""
DataAumgementer module
"""

import numpy
from ImageAugmenter import ImageAugmenter

GAUSSIAN = 'gaussian'
UNIFORM = 'uniform'

def translate(images, batch_size, channels=3, img_size=32, x_trans=(0,5), y_trans=(-5,0)):
    """
    Translates images in dataset randomly within a range in x and y directions
    
    :type images: np.array
    :param images: Array with images

    :type batch_size: int
    :param batch_size: Number of images per batch

    :type channels: int
    :param channels: Number of images per image

    :type img_size: int
    :param img_size: Size of height and with (square image)

    :type x_trans: tuple
    :param x_trans: Pixels to move in x direction

    :type y_trans: tuple
    :param y_trans: Pixels to move in y direction
    """
    if len(images.shape) == 4:
        images = (images).astype(numpy.uint8)
    elif len(images.shape) == 2:
        images = (images.reshape(batch_size,channels,img_size,img_size)).astype(numpy.uint8)
        
    augmenter = ImageAugmenter(img_size, img_size, channel_is_first_axis = True, \
                               translation_x_px=x_trans, translation_y_px=y_trans)
    tranlated_images = augmenter.augment_batch(images)
        
    return tranlated_images


def rotate(images, batch_size, channels, img_size, deg=(-5,5)):
    """
    Rotates images in dataset randomly within a limit provided
    
    :type images: np.array
    :param images: Array with images

    :type batch_size: int
    :param batch_size: Number of images per batch

    :type channels: int
    :param channels: Number of images per image

    :type img_size: int
    :param img_size: Size of height and with (square image)

    :type deg: tuple
    :param deg: Maximun rotation for image
    """
    if len(images.shape) == 4:
        images = (images).astype(numpy.uint8)
    elif len(images.shape) == 2:
        images = (images.reshape(batch_size,channels,img_size,img_size)).astype(numpy.uint8)
    
    augmenter = ImageAugmenter(img_size, img_size, channel_is_first_axis = True, rotation_deg = deg)
    rotated_images = augmenter.augment_batch(images)

    return rotated_images

def flip(images, batch_size, channels=3, img_size=32):
    """
    Flip images in dataset with 50% probability
    
    :type images: np.array
    :param images: Array with images

    :type batch_size: int
    :param batch_size: Number of images per batch

    :type channels: int
    :param channels: Number of images per image

    :type img_size: int
    :param img_size: Size of height and with (square image)
    """
    if len(images.shape) == 4:
        images = (images).astype(numpy.uint8)
    elif len(images.shape) == 2:
        images = (images.reshape(batch_size,channels,img_size,img_size)).astype(numpy.uint8)
    
    augmenter = ImageAugmenter(img_size, img_size, channel_is_first_axis = True, hflip=True)
    flipped_images = augmenter.augment_batch(images)
    
    return flipped_images

def blur(train_batch_x, noise_type, batch_size, num_channels, image_size):
    """
    Add noise to images using a Gaussian or a Normal distribution
    
    :type train_batch_x: np.array
    :param train_batch_x: Array with images

    :type noise_type: string
    :param noise_type: Type of noise to use

    :type batch_size: int
    :param batch_size: Number of images per batch

    :type channels: int
    :param channels: Number of images per image

    :type img_size: int
    :param img_size: Size of height and with (square image)
    """

    if noise_type == GAUSSIAN:
        noise = np.random.normal(            
            avg=0.0, std=0.05,
            size=(batch_size, num_channels, image_size, image_size))
    elif noise_type == UNIFORM:
        noise = np.random.uniform(low = -0.1,
                  high = 0.1,
                  size = (batch_size, num_channels, image_size, image_size))

    train_batch_x = train_batch_x + noise
    return train_batch_x
