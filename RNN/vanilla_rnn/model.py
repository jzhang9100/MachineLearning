import numpy as np
'''
Self implementation of RNN, without just using a keras submodel and then just straight up using tf.layers.LSTM
Benchmarked on Keras.IMBD data set
Network Parameters:
- U : input weight
- A : time step weight
- W : output weight -> 1D vector
- O : 1D vector -> score
'''

class s_RNN:
    def __init__(self, vocab_size, hidden_layers):
        self.vocab_size = vocab_size
        self.hidden_layers = hidden_layers
        self.U = np.random.uniform(-np.sqrt(1./vocab_size), np.sqrt(1./vocab_size), (vocab_size, hidden_layers))
        self.A = np.random.uniform(-np.sqrt(1./hidden_layers), np.sqrt(1./hidden_layers), (hidden_layers, hidden_layers))
        self.W = np.random.uniform(-np.sqrt(1./hidden_layers), np.sqrt(1./hidden_layers), (hidden_layers, hidden_layers))
        self.O = np.random.uniform(-np.sqrt(1./hidden_layers), np.sqrt(1./hidden_layers), (hidden_layers))
        print(self.U.shape, self.A.shape, self.W.shape)

    #Cross-Entropy Loss Function
    def loss(self, y_hat, y):
        if y == 1:
            return -1 * np.log(y_hat)
        else:
            return -1 * np.log(1 - y_hat)

    #BPTT implementation
    def back_prop(self, X):
        #Partial Derivatives
        d_U = np.zeros(self.U.shape)
        d_A = np.zeros(self.A.shape)
        d_W = np.zeros(self.W.shape)
        d_O = np.zeros(self.O.shape)



    def train_step(self, X, Y):
        size = len(X)

        #Time Steps
        T = []

        for i, x in enumerate(X):
            #First Block
            if i == 0:
                T.append(np.tanh(x.dot(self.U)))
            else:
                T.append(np.tanh(self.A.dot(T[i-1]) + x.dot(self.U)))
        T = np.array(T)
        out = T[size-1]
        out = out.dot(self.W)
        score = out.dot(self.O)
        loss = self.loss(score, Y)

t = []
for i in range(10):
    t.append(np.zeros(10000))
    t[i][np.random.randint(10000)] = 1
rnn = s_RNN(len(t[0]), 100)
print(rnn.train_step(t))
