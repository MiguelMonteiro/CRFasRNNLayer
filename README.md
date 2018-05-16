# CRF as RNN Layer
Conditional Random Fields as Recurrent Neural Networks (Tensorflow Implementation)

Implements Conditional Random Fields as Recurrent Neural Networks as in [the repository from the original authors.](https://github.com/sadeepj/crfasrnn_keras)
The main differences are:

1. There is CPU and a GPU kernel;

2. The code can be used with any number of spatial dimensions, input channels (number of classes) and reference channels (the original code only allows 2D and RGB reference images);

3. The GPU kernel uses the TensorFlow memory allocator, which is necessary because TensorFlow allocates most of the GPU memory by default and leaves almost none for CUDA (with large images/many channels it can become a problem);

4. The weights of the CRF are more restricted in this version so that they do not encroach on the compatibility matrix's role;

5. Support for batch_size >= 1; channel dimension is the last dimension (inputs are of the form (batch_size, ..., n_channels) with an arbitrary number of spatial dimensions in between). 

#### Compilation
To compile the code run:
````
sh build.sh
````

See the nested module [permutohedral_lattice](https://github.com/MiguelMonteiro/permutohedral_lattice) for more information on compilation for different image types.

See `Test` for dummy example usages and the [original repository](https://github.com/sadeepj/crfasrnn_keras) to see how to integrate this with a neural network (logits come in this layer and logits come out (do not feed probability or labels as inputs)).

#### Known Issues:

1. The GPU kernel allocates a fixed size hash table which uses memory proportional to the square of the number of classes. 
This might use too much memory for some applications if the number of classes is large.

2.  I have built this to use in 3D medical imaging segmentation. Even though the code works as expected, using this layer on top or a neural network or just the neural network alone didn't make a difference (statistically speaking). I don't know if its because MRI and CT images tend to have less defined edges than natural images or some other reason. If you use this and manage to get good results on medical imaging please let me know.