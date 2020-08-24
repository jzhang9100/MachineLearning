RNN

[Refrencing this Blog Post](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)

Traditional NNs have independent input/outputs. This is not very helpful when we want to do things like machine 
translation, since predicting the next word in a sentence conditions on the words that came before it.

Incomes the Recurrent Neural Network:

The RNN architecture uses previous outputs as additional inputs into the next task, so that context is provided
at each stage of learning ==> This is the idea of 'Memory' in NNs.

[RNN architecure](rnn.PNG?raw=true)

Training a RNN is very similar to a traditional NN. The difference between the two is the way backpropagation
is done. The RNN captures 'Memory' by having shared parameters across all the time steps, such that when we do
backprop, we not only have to take the gradient for the current time step but also the previous time step since
the output for the current time step is a function of the current input and the previous inputs. This process is
called Backpropagation Through Time (BPTT).
Note: 
Vanilla RNNs trained with BPTT had difficulties learning long-term dependencies (time steps far apart) due to 
vanishing/exploding gradient problem.
