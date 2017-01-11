"""
DataAumgementer module
"""

import numpy
from ImageAugmenter import ImageAugmenter



def translate_image(images, batch_size, channels=3, img_size=32, x_trans=(0,5), y_trans=(-5,0)):
    """
    """
    if len(images.shape) == 4:
        images = (images).astype(numpy.uint8)
    elif len(images.shape) == 2:
        images = (images.reshape(batch_size,channels,img_size,img_size)).astype(numpy.uint8)
    augmenter = ImageAugmenter(img_size, img_size, channel_is_first_axis = True, \
        translation_x_px=x_trans, translation_y_px=y_trans)
    tranlated_images = augmenter.augment_batch(images)
        
    return tranlated_images


def rotate_image(images, batch_size, channels, img_size, deg=(-5,5)):
    """
    """
    if len(images.shape) == 4:
        images = (images).astype(numpy.uint8)
    elif len(images.shape) == 2:
        images = (images.reshape(batch_size,channels,img_size,img_size)).astype(numpy.uint8)
    augmenter = ImageAugmenter(img_size, img_size, channel_is_first_axis = True, rotation_deg = deg)
    rotated_images = augmenter.augment_batch(images)

    return rotated_images

def flip_image(images, batch_size, channels=3, img_size=32):

    if len(images.shape) == 4:
        images = (images).astype(numpy.uint8)
    elif len(images.shape) == 2:
        images = (images.reshape(batch_size,channels,img_size,img_size)).astype(numpy.uint8)
    augmenter = ImageAugmenter(img_size, img_size, channel_is_first_axis = True, hflip=True)
    flipped_images = augmenter.augment_batch(images)
    
    return flipped_images

def noise_injection(train_batch_x, noise_type, batch_size, num_channels, image_size):
    """
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
