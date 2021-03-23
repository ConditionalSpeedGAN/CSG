# Implementation Details
### Feature Extractor: 
We pass the inputs of size 3 and 6 for single and multi-agent scenarios respectively through a fixed size embedding layer of 16 units to match the input shape of the LSTMS with 32 dimensions. The encoder is comprised of LSTM cells, one for each agent that unroll until the length of the observed timesteps.

### Aggregation Module 1: (Pooling Module)
The relative positions of the agents are embedded by passing through a single layer FC with 16 units. The hidden states and the embedded relative positions are then concatenated and passed through a multi layer MLP of dimensions 48, 512, 32 where the first dim represents the concatenated input of h\_states and embedded rel. positions and the final unit represents the bottleneck dimension. The output of FC network is passed through a Max pooling layer.

### Aggregation Module 2: (Attention Module)
The methodology for attention module used is similar to PM with slight difference. We consider 'N' nearest agents and embed their respective relative positions and the hidden states. Additionally, the max pooling is replaced with single layer FC network followed by softmax activation that denotes the attention weights for the N considered agents. The single FC network in attention block consists of 64*N input units and 'N' output units.

### Aggregation Module 3: (Concat Module)
Similar to attention module, we consider 'N' nearest agents and pass the considered hidden states of the neighbours through a multi layer FC network of dimension  32*N, 512 and 32.  

### No Aggregation Module 4:
The final hidden states of the Encoder LSTM are retrieved. 

Finally, the final hidden states are then embedded by passing through a multi layer FC with 32,64 and 24 units before adding noise of size 8. We replace the first dimension as 64 if aggregation component is used where 32 represents encoder hidden states and the rest 32 indicates the aggregation latent space units. We use these latent vectors to initialise the speed forecaster and the decoder.

### Speed Forecasting Block: 
The speed forecasting block uses a single layer of LSTMs with 32 dimensions. The input, which is the last observed speed, is passed through an embedding layer of 16 dimensions before being fed to LSTMs with 32 hidden units. The final hidden states are then passed through a FC with 32 and 1 units followed by sigmoid activations, to represent the speed at the next time step.

### Decoder: 
Similar to the feature extractor, we embed our inputs to a fixed size vector. The hidden states of the decoder are passed through a FC network of 32 and 2 units with the outputs representing the next timestep relative coordinates.

### Discriminator: 
In discriminator, the encoder LSTM uses 1 hidden layer and 64 hidden dimensions. The final hidden state of the encoder in discriminator is passed through a MLP of dimension 64, 1028 and 1 where the final dimension represents the classification score.


We train the GAN and simultaneously train the speed forecaster. During prediction of pedestrians we condition the trajectory decoder with values obtained from the speed forecaster. During multi-agent prediction, we also use the class labels and condition the latent space and the decoder with it. During simulation, we use user defined values as the condition.

The learning rate is set to 0.001 with Adam optimizer and the mode is trained for 50 epochs without aggregation and 100 epochs with aggregation using 16 and 32 as mini batch for single and multi conditions respectively. 
