
# coding: utf-8

# In[1]:


from __future__ import print_function

import sys
import pickle
import timeit
import numpy as np
from random import Random

import theano
import theano.tensor as T


from helper import data_augmenter
from helper.metrics import nn_score
from helper.optimizer import SGD, Adam
from helper.operator import batchnorm, deconv
from helper.initialization import normal, constant
from helper.data_augmenter import translate, rotate, flip, blur
from helper.network_builder import build_discriminator, build_generator
from helper.image_manipulation import transform, inverse_transform, generate_samples
from helper.data_handler import load_dataset, concatenate_datasets, create_dataset_from_folder

relu = T.nnet.relu
tanh = T.tanh

binary_crossentropy = T.nnet.binary_crossentropy

theano.config.floatX = 'float32'

rng = np.random.RandomState(42)

f = open('log.txt','w')


def main():
    # # Initial Parameters

    # In[2]:

    # Optimizer parameters
    l2_reg = 1e-5               # l2_reg weight decay
    learning_rate = 0.0002      # initial learning rate for adam
    momentum = 0.5              # momentum term of adam

    # Image dimension
    n_channels = 3              # Number of channels in image
    img_size = 64               # Number of pixels width/height of images

    # Training parameters
    n_gen_samples = 196         # Number of samples to save during
    n_epochs = 1000             # Number of epochs
    batch_size = 128            # Number of examples in batch
    epoch_results = 1           # Frequency to save the intermediate results
    epoch_params = 1           # Frequency to save the network parameters

    #Architecture parameters
    n_g_filters = 128           # Number of generator filters in first conv layer
    n_d_filters = 128           # Number of discriminator filters in first conv layer
    dimZ = 100                  # Number of dim for Z


    # ### Network architecture

    # In[3]:

    # Theano variable for real images
    X = T.tensor4('X')
    # Theano variable for random noise vector
    Z = T.matrix('Z')

    # Generator architecture
    g_initial_im_size = 4
    g_flat_size = (dimZ, n_g_filters*8*4*4)
    g_layer_size = [n_g_filters*4, n_g_filters*2, n_d_filters, n_channels]
    g_num_filters = [n_g_filters*8, n_g_filters*4, n_d_filters*2, n_d_filters]
    g_filter_size = [5, 5, 5, 5]
    g_norm = [True, True, True, False]
    g_activation = [relu, relu, relu, tanh]
    g_subsample = [(2,2),(2,2),(2,2),(2,2)]
    g_border_mode = [(2,2),(2,2),(2,2),(2,2)]

    # Discriminator architecture
    d_layer_size = [n_channels, n_d_filters, n_d_filters*2, n_d_filters*4]
    d_num_filters = [n_d_filters, n_d_filters*2, n_d_filters*4, n_d_filters*8]
    d_filter_size = [5, 5, 5, 5]
    d_norm = [False, True, True, True]
    d_flat_size = (n_d_filters*8*4*4, 1)


    # # Load data

    # In[4]:

    # Load training and testing dataset and combine them to have a bigger sample
    cars, n_train_batches = load_dataset('../data/cars.npy', batch_size)
    cars2, n_train_batches = load_dataset('../data/cars_test.npy', batch_size)
    datasetX, n_train_batches = concatenate_datasets((cars, cars2), batch_size, axis=0)
    datasetX = transform(datasetX)
    
    # The same random vector is used each time to generate the samples stored
    # The reason to do this lies on being able to see the evolution from a set of inputs over time
    sample_randomZ = np.asarray(rng.uniform(-1., 1., size=(n_gen_samples, dimZ)), dtype=theano.config.floatX)

    # # Build Network

    # In[5]:

    # Build generator
    gX, g_layers = build_generator(Z, g_layer_size, g_num_filters, g_filter_size, g_flat_size, 
                                   g_subsample, g_border_mode, g_norm, g_activation, g_initial_im_size)

    # List of generator parameters to be optimized
    g_params = [item for sublist in [layer.params for layer in g_layers] for item in sublist]

    # Build discriminator for real samples
    p_real, d_real_layers = build_discriminator(X, d_layer_size, d_num_filters, d_filter_size, 
                                                d_flat_size, d_norm)

    # List of discriminator parameters to be optimized
    d_params = [item for sublist in [layer.params for layer in d_real_layers] for item in sublist]

    # Build discriminator for generated samples, it is the same as the previous discriminator
    # They are duplicate to facilitate creating the assignid the label for the real and false
    # samples. 
    p_gen, d_gen_layers = build_discriminator(gX, d_layer_size, d_num_filters, d_filter_size, 
                                              d_flat_size, d_norm, d_params)

    # Cost to be optimized
    # The cost of the generator is the sum of the loss of not predicting the real images as one
    # plus the loss of not predicting the false images as zeros. The cost of the generator 
    # is the loss of not being able to fool the discriminator
    d_cost_real = binary_crossentropy(p_real, T.ones(p_real.shape)).mean()
    d_cost_gen = binary_crossentropy(p_gen, T.zeros(p_gen.shape)).mean()
    g_cost_d = binary_crossentropy(p_gen, T.ones(p_gen.shape)).mean()

    d_cost = d_cost_real + d_cost_gen
    g_cost = g_cost_d

    lr_shared = theano.shared(np.asarray(learning_rate, dtype=theano.config.floatX))
    d_updater = Adam(lr=lr_shared, b1=momentum, l2=l2_reg)
    g_updater = Adam(lr=lr_shared, b1=momentum, l2=l2_reg)

    d_updates = d_updater(d_params, d_cost)
    g_updates = g_updater(g_params, g_cost)
    updates = d_updates + g_updates

    # The theano function using the defines cost and the update procedure are created
    _train_g = theano.function([Z], g_cost, updates=g_updates)
    _train_d = theano.function([X, Z], d_cost, updates=d_updates)
    _gen = theano.function([Z], gX)

    # In[7]:

    # Only one pass over the entire dataset for the discriminator
    # is done in this dataset before training both networks
    for minibatch_index in range(n_train_batches):

        x_batch = datasetX[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]        
        randomZ = np.asarray(rng.uniform(-1., 1., size=(len(x_batch), dimZ)), dtype=theano.config.floatX)
        d_loss = _train_d(x_batch, randomZ)

    print(('d_cost %.4f') % (d_loss), file=f)        



    # In[ ]:

    epoch = 0    

    start_time = timeit.default_timer()

    while (epoch < n_epochs):

        # Shuffle samples at the beginning of each epoch
        np.random.shuffle(datasetX)

        # For each minibatch on the training set
        for minibatch_index in range(3):#n_train_batches):

            # A minibath is retrived and a random vector Z is sampled
            x_batch = datasetX[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]        
            randomZ = np.asarray(rng.uniform(-1., 1., size=(len(x_batch), dimZ)), dtype=theano.config.floatX)

            # The generator is trained in the first place 
            g_loss = _train_g(randomZ)   

            # Afterwards, the discriminator is trained on a new random vector
            randomZ = np.asarray(rng.uniform(-1., 1., size=(len(x_batch), dimZ)), dtype=theano.config.floatX)
            d_loss = _train_d(x_batch, randomZ)   

            # Only use in some test cases where the cost of the generator diverges quickly
            #if (g_loss-1.5) > d_loss:
            #    randomZ = np.asarray(rng.uniform(-1., 1., size=(len(x_batch), dimZ)), dtype=theano.config.floatX)
            #    g_loss = _train_g(randomZ)        

        print(('epoch %d g_loss %.4f d_loss %.4f') % (epoch, g_loss, d_loss), file=f)    
        f.flush()    

        # Validation metric is calculated by obtained the distance to the closet real
        # sample of the generated ones using 1 Nearest Neighbour
        if epoch % epoch_results == 0:
            gX = generate_samples(_gen, n_batches=1, batch_size=batch_size, size_Z=dimZ)
            gX = gX.reshape(len(gX), -1)

            validation = nn_score(gX, x_batch.reshape(len(x_batch), -1))    
            print(('validation %.2f g_loss %.4f d_loss %.4f') % (validation, g_loss, d_loss), file=f)    
            f.flush()

            np.save('../results/new_generated_images' + str(epoch) + '.npy', _gen(sample_randomZ))  
            
        if epoch % epoch_params == 0:
            pickle.dump(d_params, open('../results/d_params.p', 'wb'))
            pickle.dump(g_params, open('../results/g_params.p', 'wb'))
            pickle.dump(sample_randomZ, open('../results/sample_randomZ.p', 'wb'))

        epoch = epoch + 1    

    pickle.dump(d_params, open( "d_params.p", "wb" ))
    pickle.dump(g_params, open( "g_params.p", "wb" ))
    pickle.dump(sample_randomZ, open( "sample_randomZ.p", "wb" ))

    print(('The code ran for %.2fm' % ((end_time - start_time) / 60.)), file=f)    
    f.flush()
if __name__ == '__main__':
    main()