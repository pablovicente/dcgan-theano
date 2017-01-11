"""
data_handler module
"""

import os
import sys
import tarfile
import numpy as np
from PIL import Image
from scipy import misc, io

def load_dataset(path, batch_size):
    """
    Load a numpy array that stores dataset samples
    
    :type partial_datasets: string
    :param partial_datasets: Path to dataset file

    :type batch_size: int
    :param batch_size: Size of the batch size to be used
    """
    
    assert path is not None, 'Provide a valid dataset'
    
    dataset = np.load(path)

    n_batches = len(dataset)
    n_batches //= batch_size
    
    return dataset, n_batches

def concatenate_datasets(partial_datasets, batch_size, axis=0):
    """
    Contatenates several numpy arrays in one to form 
    a bigger dataset array
    
    :type partial_datasets: tuple
    :param partial_datasets: Arrays to be concatenated

    :type batch_size: int
    :param batch_size: Size of the batch size to be used
    
    :type axis: int
    :param axis: Axis from which to concatenate
    """
    
    assert isinstance(partial_datasets, tuple), 'Datasets must be in a tuple'
    
    dataset = np.concatenate(partial_datasets, axis=axis)
    
    n_batches = len(dataset)
    n_batches //= batch_size

    return dataset, n_batches

def create_dataset_from_compressed_file(path_origin, path_destination, file_format='jpg'):
    """
    Load images from a compressed file and store them 
    in a numpy array to be saved in memory
    
    :type path_origin: string
    :param path_origin: path where the images are contained

    :type path_destination: string
    :param path_destination: destination path for the numpy array created
    
    :type file_format: string
    :param file_format: file formats accepted
    """    
        
    # Decompress the file 
    tar = tarfile.open(path_origin)
    file_names = tar.getnames()
    for file_name in file_names:
        tar.extract(file_name)
    tar.close()       

    # Removes the alpha channel of the images 
    # and overrides them
    images = []
    for file_name in file_names:
        if file_format in file_name:
            try:
                image = Image.open(file_name)
                image.load() 

                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3]) # 3 is the alpha channel
                background.save(file_name, 'JPEG', quality=100)

            except IOError:
                pass

    # Stores images in a list         
    images = []
    for file_name in file_names:
        if file_format in file_name:
            try:
                images.append(misc.imread(file_name))

            except IOError:
                pass

    if len(images) > 0:
        # Set channel in the second index        
        image_array = np.asarray(images).transpose(0,3,1,2)
        # Saves images in as a numpy array     
        np.save(path_destination, image_array)
    
    print('%d images were processed' % len(images))    

def create_dataset_from_folder(path_origin, path_destination, file_format='jpg'):
    """
    Load images from a folder and store them 
    in a numpy array to be saved in memory
    
    :type path_origin: string
    :param path_origin: path where the images are contained

    :type path_destination: string
    :param path_destination: destination path for the numpy array created
    
    :type file_format: string
    :param file_format: file formats accepted
    """
    
    # Load images filenames
    file_names = []
    path = path_origin
    for (dirpath, dirnames, filenames) in os.walk(path):
        file_names.extend(filenames)
        break


    # Looad images files and saved them in a list
    images = []
    for name in file_names:
        if file_format in name:
            try:
                image = misc.imread(path + name)
                if image.shape == (64,64,3):
                    images.append(image)
            except IOError:
                pass

      
    # Convert the list to an array and save dataset to disk
    if len(images) > 0:
        image_array = np.asarray(images).transpose(0,3,1,2)
        np.save(path_destination, image_array)
    
    print('%d images were processed' % len(images))