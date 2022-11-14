import numpy as np
import tensorflow as tf
import time
from abc import ABC, abstractmethod

class FBSNN(ABC): # Forward-Backward Stochastic Neural Network
    def __init__(self, Xi, T, M, N, D, layers):
        
        self.Xi = Xi # initial point
        self.T = T # terminal time
        
        self.M = M # number of trajectories
        self.N = N # number of time snapshots
        self.D = D # number of dimensions
        
        # layers
        self.layers = layers # (D+1) --> 1
        
        # initialize NN
        self.model = FBSNNModel(layers)

        # optimizers
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LrScheduler(), epsilon=1e-8)

    def net_u(self, t, X): # M x 1, M x D
        u = self.model(tf.concat([t, X], 1)) # M x 1
        Du = tf.gradients(u, X) # M x D
        return u, Du[0]

    def Dg_tf(self, X): # M x D
        g = self.g_tf(X)
        return tf.gradients(g, X)[0] # M x D
    
    @tf.function
    def loss_function(self, t, W, Xi): # M x (N+1) x 1, M x (N+1) x D, 1 x D
        loss = tf.constant(0.)
        X_list = tf.TensorArray(tf.float32, size=self.N + 1)
        Y_list = tf.TensorArray(tf.float32, size=self.N + 1)
        
        t0 = t[:,0,:]
        W0 = W[:,0,:]
        X0 = tf.tile(Xi,[self.M,1]) # M x D

        Y0, Z0 = self.net_u(t0, X0) # M x 1, M x D
    
        X_list = X_list.write(0, X0)
        Y_list = Y_list.write(0, Y0)

        for n in tf.range(0, self.N):
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]
            
            X0, Y0, Z0, l = self.forward(t0, t1, W0, W1, X0, Y0, Z0)
            t0 = t1
            W0 = W1
            loss += l
            
            X_list = X_list.write(n + 1, X0)
            Y_list = Y_list.write(n + 1, Y0)
        
        loss += tf.reduce_sum(tf.square(Y0 - self.g_tf(X0)))
        loss += tf.reduce_sum(tf.square(Z0 - self.Dg_tf(X0)))
        
        X = tf.transpose(X_list.stack(), [1, 0, 2])
        Y = tf.transpose(Y_list.stack(), [1, 0, 2])

        return loss, X, Y, Y[0,0,0]

    def forward(self, t0, t1, W0, W1, X0, Y0, Z0):
        X1 = X0 + self.mu_tf(t0,X0,Y0,Z0)*(t1-t0) + tf.squeeze(tf.matmul(self.sigma_tf(t0,X0,Y0), tf.expand_dims(W1-W0,-1)), axis=[-1])
        Y1_tilde = Y0 + self.phi_tf(t0,X0,Y0,Z0)*(t1-t0) + tf.reduce_sum(Z0*tf.squeeze(tf.matmul(self.sigma_tf(t0,X0,Y0), tf.expand_dims(W1-W0, -1))), axis=1, keepdims=True)
        Y1, Z1 = self.net_u(t1, X1)
        loss = tf.reduce_sum(tf.square(Y1 - Y1_tilde))
        return X1, Y1, Z1, loss

    def fetch_minibatch(self):
        T = self.T
        
        M = self.M
        N = self.N
        D = self.D
        
        Dt = np.zeros((M,N+1,1)) # M x (N+1) x 1
        DW = np.zeros((M,N+1,D)) # M x (N+1) x D
        
        dt = T/N
        
        Dt[:, 1:, :] = dt
        DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(M,N,D))
        
        t = np.cumsum(Dt, axis=1) # M x (N+1) x 1
        W = np.cumsum(DW, axis=1) # M x (N+1) x D
        t = t.astype(np.float32)
        W = W.astype(np.float32)

        return t, W
    
    def train(self, N_Iter):
        
        start_time = time.time()
        for it in range(N_Iter):     
            t_batch, W_batch = self.fetch_minibatch() # M x (N+1) x 1, M x (N+1) x D
            
            with tf.GradientTape() as t:
                loss, X_pred, Y_pred, Y0_pred = self.loss_function(t_batch, W_batch, self.Xi)
            grads = t.gradient(loss, self.model.variables)
            self.optimizer.apply_gradients(grads_and_vars=zip(grads, self.model.variables))

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f' % (it, loss, Y0_pred, elapsed))
                start_time = time.time()
                
    
    def predict(self, Xi_star, t_star, W_star):
        loss, X_pred, Y_pred, Y0_pred = self.loss_function(t_star, W_star, Xi_star)
        return X_pred, Y_pred

    @abstractmethod
    def phi_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        pass # M x 1
    
    @abstractmethod
    def g_tf(self, X): # M x D
        pass # M x 1
    
    @abstractmethod
    def mu_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        M = self.M
        D = self.D
        return tf.zeros([M,D]) # M x D
    
    @abstractmethod
    def sigma_tf(self, t, X, Y): # M x 1, M x D, M x 1
        M = self.M
        D = self.D
        return tf.linalg.diag(tf.ones([M,D])) # M x D x D
    ###########################################################################

class FBSNNModel(tf.keras.Model):
    def __init__(self, layers):
        super().__init__()
        self.dense_layers = []
        for i in range(len(layers) - 2):
            layer = tf.keras.layers.Dense(
                units=layers[i + 1],
                activation=tf.math.sin,
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                bias_initializer=tf.zeros_initializer(),
            )
            self.dense_layers.append(layer)
        layer = tf.keras.layers.Dense(
                units=layers[-1],
                activation=None,
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                bias_initializer=tf.zeros_initializer(),
            )
        self.dense_layers.append(layer)
    
    def call(self, input):
        x = input
        for layer in self.dense_layers:
            x = layer(x)
        return x

class LrScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __call__(self, step):
        if step < 20000:
            return 1e-3
        elif step < 50000:
            return 1e-4
        elif step < 80000:
            return 1e-5
        else:
            return 1e-6