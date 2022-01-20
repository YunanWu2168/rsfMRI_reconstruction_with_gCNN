import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import scipy.sparse
import scipy.sparse.linalg
import tensorflow as tf
from tensorflow.python.ops import array_ops
import tensorflow_probability as tfp
#from vis import utils
#from vis.utils import utils

def corr_coef(y_true, y_pred):
    return tfp.stats.correlation(y_true, y_pred)

def gmse(y_true, y_pred): #MSE for graph auto_encoder
    mask = K.cast(K.not_equal(y_true, 0), K.floatx())
    y_pred = y_pred * mask
    gmse = K.square(y_true - y_pred)
    gmse = K.sum(gmse, axis=(1,2)) / K.sum(mask, axis=(1,2))
    return K.log(gmse)

def gmae(y_true, y_pred): #MAE for graph auto_encoder
    mask = K.cast(K.not_equal(y_true, 0), K.floatx())
    y_pred = y_pred * mask
    gmae = K.abs(y_true - y_pred)
    gmae = K.sum(gmae, axis=(1,2)) / K.sum(mask, axis=(1,2))
    return gmae

def log_mse(y_true, y_pred): #Log MSE loss
    gmse = K.square(y_true - y_pred)
    gmse = K.mean(gmse) + K.epsilon()
    return K.log(gmse)

def exp_mse(y_true, y_pred):
    emse = 10 * K.abs(y_true - y_pred) / y_true
    emse = K.exp(emse) - 1
    return K.mean(emse)

def log_mare(y_true, y_pred): # Log of mean absolute relative error
    lmare = 10 * K.square(y_true - y_pred) / y_true
    lmare = K.mean(lmare) + K.epsilon()
    return K.log(lmare)

def mean_log(y_true, y_pred): #Log MSE loss
    gmse = K.square(y_true - y_pred) + K.epsilon()
    return K.log(gmse)

def mse_corr(y_true, y_pred):
    gmse = K.square(y_true - y_pred)
    gmse = K.mean(gmse) + K.epsilon()
    corr = corr_coef(y_true, y_pred - y_true)
    corr = 2 * K.relu(-corr, alpha=0.001)
    return K.log(gmse) + corr
    #mcorr = K.square(y_true - y_pred)
    #mcorr = K.mean(mcorr) + K.epsilon()
    #mcorr = K.mean(K.square(y_true - y_pred))
    #corr = corr_coef(y_true, y_pred - y_true)
    #return K.log(mcorr + 2*K.relu(-corr, alpha=0.1) + K.epsilon())
    #return mcorr + 0.01 * (1 - corr_coef(y_true, y_pred))

def f1_score(y_true, y_pred): #taken from old keras source code
    if K.ndim(y_true) == K.ndim(y_pred):
        y_true = K.squeeze(y_true, -1)
    y_pred_labels = K.argmax(y_pred, axis=-1)
    y_pred = K.cast(y_pred_labels, K.floatx())
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def matth_corr(y_true, y_pred):
    if K.ndim(y_true) == K.ndim(y_pred):
        y_true = K.squeeze(y_true, -1)
    y_pred_labels = K.argmax(y_pred, axis=-1)
    y_pred_labels = K.cast(y_pred_labels, K.floatx())
    y_pred_pos = K.cast(K.equal(y_pred_labels, 1), K.floatx())
    y_pred_neg = K.cast(K.equal(y_pred_labels, 0), K.floatx())

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = K.cast(K.equal(y_pos, 0), K.floatx())

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

def matth_corr_sp(y_true, y_pred):
    if K.ndim(y_true) == K.ndim(y_pred):
        y_true = K.squeeze(y_true, -1)
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = K.cast(K.equal(y_pred_pos, 0), K.floatx())

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = K.cast(K.equal(y_pos, 0), K.floatx())

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

def kslice(start):
    def func(x):
        return x[:,start::2,:]
    return keras.layers.Lambda(func)


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, labels=None, batch_size=32, shuffle=True, force_balance=False, pick_weights=None, do_augmentation=False, tta=1, aug_params=None, auto_encoder=False, training=False):
        'Initialization'
        self.batch_size = batch_size
        self.data = data
        self.labels = labels
        self.shuffle = shuffle
        self.force_balance = force_balance
        self.do_augmentation = do_augmentation
        self.tta = tta
        self.auto_encoder = auto_encoder
        self.pick_weights = pick_weights
        self.training = training

        if self.auto_encoder:
            self.force_balance = False

        # Augmentation parameters
        if aug_params is None:
            self.max_rotation = 60
            self.max_zoom = 0.05
            self.max_noise = 0.0001
            self.max_shift = 0.0005
            self.prob_augment = 1.
        else:
            self.max_rotation = aug_params["max_rotation"]
            self.max_zoom = aug_params["max_zoom"]
            self.max_noise = aug_params["max_noise"]
            self.max_shift = aug_params["max_shift"]
            self.prob_augment = aug_params["prob_augment"]

        # If force_balance, pre-compute picking weights
        if self.force_balance:
            if pick_weights is None:
                self.pick_weights = np.zeros(labels.shape)
                unique_label, counts_label = np.unique(labels, return_counts = True)
                for i, lab in enumerate(unique_label):
                    self.pick_weights[labels == lab] = 1 / counts_label[i]

                self.pick_weights = self.pick_weights / np.sum(self.pick_weights)
            else:
                self.pick_weights = pick_weights

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.training == False:
            return int(np.ceil(self.data.shape[0] * self.tta / self.batch_size))
        else:
            return int(np.floor(self.data.shape[0] * self.tta / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if not self.force_balance:
            self.indexes = np.arange(self.data.shape[0])
            if self.shuffle == True:
                np.random.shuffle(self.indexes)
        else:
            self.indexes = np.random.choice(np.arange(self.data.shape[0]), size = (self.data.shape[0],), p = self.pick_weights)

        if self.tta > 1:
            self.indexes = np.tile(self.indexes, self.tta)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((len(indexes), self.data.shape[1], self.data.shape[2]))
        if not self.auto_encoder:
                y = np.empty((len(indexes)), dtype=self.labels.dtype)

        # Generate data
        for i, ID in enumerate(indexes):
            # Store sample
            X[i,] = self.data[ID,]

            # Store class
            if not self.auto_encoder:
                y[i] = self.labels[ID,]

        if self.do_augmentation:
            X = self.augment_data(X, max_rotation=self.max_rotation,
                                  max_zoom=self.max_zoom,
                                  max_noise=self.max_noise,
                                  max_shift=self.max_shift,
                                  prob_augment=self.prob_augment)

        if not self.auto_encoder:
            return X, y
        else:
            return X, X

class DataGenerator2(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, labels=None, batch_size=32, shuffle=True, force_balance=False, pick_weights=None, do_augmentation=False, tta=1, aug_params=None, auto_encoder=False, training=False):
        'Initialization'
        self.batch_size = batch_size
        self.data = data
        self.labels = labels
        self.shuffle = shuffle
        self.force_balance = force_balance
        self.do_augmentation = do_augmentation
        self.tta = tta
        self.auto_encoder = auto_encoder
        self.pick_weights = pick_weights
        self.training = training

        if self.auto_encoder:
            self.force_balance = False

        # Augmentation parameters
        if aug_params is None:
            self.max_rotation = 60
            self.max_zoom = 0.05
            self.max_noise = 0.0001
            self.max_shift = 0.0005
            self.prob_augment = 1.
        else:
            self.max_rotation = aug_params["max_rotation"]
            self.max_zoom = aug_params["max_zoom"]
            self.max_noise = aug_params["max_noise"]
            self.max_shift = aug_params["max_shift"]
            self.prob_augment = aug_params["prob_augment"]

        # If force_balance, pre-compute picking weights
        if self.force_balance:
            if pick_weights is None:
                self.pick_weights = np.zeros(labels.shape)
                unique_label, counts_label = np.unique(labels, return_counts = True)
                for i, lab in enumerate(unique_label):
                    self.pick_weights[labels == lab] = 1 / counts_label[i]

                self.pick_weights = self.pick_weights / np.sum(self.pick_weights)
            else:
                self.pick_weights = pick_weights

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.training == False:
            return int(np.ceil(self.data.shape[0] * self.tta / self.batch_size))
        else:
            return int(np.floor(self.data.shape[0] * self.tta / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if not self.force_balance:
            self.indexes = np.arange(self.data.shape[0])
            if self.shuffle == True:
                np.random.shuffle(self.indexes)
        else:
            self.indexes = np.random.choice(np.arange(self.data.shape[0]), size = (self.data.shape[0],), p = self.pick_weights)

        if self.tta > 1:
            self.indexes = np.tile(self.indexes, self.tta)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((len(indexes), self.data.shape[1], self.data.shape[2], self.data.shape[3]))
        if not self.auto_encoder:
                y = np.empty((len(indexes)), dtype=self.labels.dtype)

        # Generate data
        for i, ID in enumerate(indexes):
            # Store sample
            X[i,] = self.data[ID,]

            # Store class
            if not self.auto_encoder:
                y[i] = self.labels[ID,]

        if self.do_augmentation:
            X = self.augment_data(X, max_rotation=self.max_rotation,
                                  max_zoom=self.max_zoom,
                                  max_noise=self.max_noise,
                                  max_shift=self.max_shift,
                                  prob_augment=self.prob_augment)

        if not self.auto_encoder:
            return X, y
        else:
            return X, X


    def augment_data(self, X, max_rotation = 60, max_zoom = 0.1, max_noise = 0.00, max_shift = 0.005, prob_augment = 1.):
        """
        Do data augmentation on the surface
        augment_data(x, augment_params)

        x [batch_size, N_nodes, N_coordinates]
           Cartesian coordinates of the surface vertices.
           N_coordinates can be 3, 6 or 9.

        """

        if prob_augment == 0:
            return X

        max_angle_radian = np.radians(max_rotation)

        # Determine transformation parameters
        r_xyz = (np.random.random([X.shape[0], 3]) - 0.5) * max_angle_radian
        h_factor = 1.0 - (np.random.random([X.shape[0]]) - 0.5) * max_zoom
        r_shift = (np.random.random([X.shape[0], 1, 3, 1]) - 0.5) * max_shift
        do_augment = np.random.random([X.shape[0]]) < prob_augment
        r_x   = np.zeros((X.shape[0], 1, 3, 3))
        r_y   = np.zeros((X.shape[0], 1, 3, 3))
        r_z   = np.zeros((X.shape[0], 1, 3, 3))
        H     = np.zeros((X.shape[0], 1, 3, 3))

        # Fills transformation matrices
        r_x[:,0,0,0] = 1;                   r_x[:,0,0,1] = 0;                   r_x[:,0,0,2] = 0
        r_x[:,0,1,0] = 0;                   r_x[:,0,1,1] = np.cos(r_xyz[:,0]);  r_x[:,0,1,2] = -np.sin(r_xyz[:,0])
        r_x[:,0,2,0] = 0;                   r_x[:,0,2,1] = np.sin(r_xyz[:,0]);  r_x[:,0,2,2] = np.cos(r_xyz[:,0])
        r_y[:,0,0,0] = np.cos(r_xyz[:,1]);  r_y[:,0,0,1] = 0;                   r_y[:,0,0,2] = np.sin(r_xyz[:,1])
        r_y[:,0,1,0] = 0;                   r_y[:,0,1,1] = 1;                   r_y[:,0,1,2] = 0
        r_y[:,0,2,0] = -np.sin(r_xyz[:,1]); r_y[:,0,2,1] = 0;                   r_y[:,0,2,2] = np.cos(r_xyz[:,1])
        r_z[:,0,0,0] = np.cos(r_xyz[:,2]);  r_z[:,0,0,1] = -np.sin(r_xyz[:,2]); r_z[:,0,0,2] = 0
        r_z[:,0,1,0] = np.sin(r_xyz[:,2]);  r_z[:,0,1,1] = np.cos(r_xyz[:,2]);  r_z[:,0,1,2] = 0
        r_z[:,0,2,0] = 0;                   r_z[:,0,2,1] = 0;                   r_z[:,0,2,2] = 1
        H[:,0,0,0] = h_factor[:];           H[:,0,1,1] = h_factor[:];           H[:,0,2,2] = h_factor[:]
        x_mat = np.matmul(r_x, r_y); x_mat = np.matmul(x_mat, r_z); x_mat = np.matmul(x_mat, H)
        # Rx = [1 0 0; 0 cos(t_x) -sin(t_x); 0 sin(t_x) cos(t_x)];
        # Ry = [cos(t_y) 0 sin(t_y); 0 1 0; -sin(t_y) 0 cos(t_y)];
        # Rz = [cos(t_z) -sin(t_z) 0; sin(t_z) cos(t_z) 0; 0 0 1];
        # H = [h_factor 0 0; 0 h_factor 0; 0 0 h_factor];

        # Reshape, transpose, apply XMats and add shifts
        X_out = np.reshape(X, (X.shape[0], X.shape[1], 3, int(X.shape[2]/3)))
        X_out = np.transpose(X_out, axes=(0, 1, 3, 2))
        X_out = np.matmul(x_mat, X_out)
        X_out += r_shift
        X_out = np.transpose(X_out, axes=(0, 1, 3, 2))
        X_out = np.reshape(X_out, (X.shape[0], X.shape[1], -1))

        # Add noise
        if max_noise != 0:
            X_out += np.random.normal(loc=0, scale=max_noise, size=X_out.shape)

        # Replace non-augmented instance with the original ones
        X_out[do_augment == 0,:,:] = X[do_augment == 0,:,:]

        # Reset the zeros
        X_out[X==0] = 0

        return X_out


class ChebLayer(keras.layers.Layer):

    def __init__(self, L,
                 rank,
                 filters,
                 level = None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 **kwargs):
        super(ChebLayer, self).__init__(**kwargs)
        if scipy.sparse.issparse(L):
            self.L = L
            self.L_dense = scipy.sparse.find(L)
        else:
            self.L = scipy.sparse.csr_matrix((L[2], (L[0], L[1])), shape=(max(L[0])+1, max(L[0])+1), dtype=np.float32)
            self.L_dense = L
        #self.L = L
        self.rank = int(rank)
        self.filters = int(filters)
        self.level = level
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_dim  = int(input_shape[-1])
        output_dim = self.filters
        self.kernel = self.add_weight(name='kernel', shape=(input_dim * self.rank, self.filters), initializer=self.kernel_initializer, regularizer=self.kernel_regularizer, trainable=True)
        # Create trainable biases
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer)
        else:
            self.bias = None


        # Rearange or load Laplacian
        with tf.compat.v1.variable_scope("Level_"+str(self.level), reuse=tf.compat.v1.AUTO_REUSE):
            # Define rescale_L
            def rescale_L(L, lmax=2, scale=1):
                """Rescale the Laplacian eigenvalues in [-1,1]."""
                M, M = L.shape
                I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
                L *= 2 * scale / lmax
                L -= I
                return L
            with tf.device('CPU:0'):
                L = scipy.sparse.csr_matrix(self.L)
                lmax = 1.02*scipy.sparse.linalg.eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]
                L = rescale_L(L, lmax=lmax, scale=0.75)
                L = L.tocoo()
                indices = np.column_stack((L.row, L.col))
                L = tf.SparseTensor(indices, L.data, L.shape)
                L = tf.sparse.reorder(L)
                self.L_tf = L

        super(ChebLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        N, M, Fin = x.shape # batch_size x number_nodes x number_features
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, -1])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if self.rank > 1:
            x1 = tf.sparse.sparse_dense_matmul(self.L_tf, x0)
            x = concat(x, x1)
        for k in range(2, self.rank):
            x2 = 2 * tf.sparse.sparse_dense_matmul(self.L_tf, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [self.rank, M, Fin, -1])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [-1, Fin*self.rank])  # N*M x Fin*K
        x = tf.matmul(x, self.kernel)
        if self.use_bias:
            x = x + self.bias # N*M x Fout
        return tf.reshape(x, [-1, M, self.filters])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.filters)

    def get_config(self):
        config = super(ChebLayer, self).get_config()
        config.update({'L': self.L_dense,
                       'rank': self.rank,
                       'filters': self.filters,
                       'level': self.level,
                       'use_bias': self.use_bias,
                       'kernel_initializer': self.kernel_initializer,
                       'bias_initializer': self.bias_initializer,
                       'kernel_regularizer': self.kernel_regularizer,
                       'bias_regularizer': self.bias_regularizer,
                       'activity_regularizer': self.activity_regularizer})
        return config

def get_cam(model, input_, class_idx=0):
    ## feature map from the final convolusional layer
    final_fmap_index    = utils.find_layer_idx(model, 'dense')
    penultimate_output  = model.layers[final_fmap_index].output

    ## define derivative d loss^c / d A^k,k =1,...,512
    layer_input          = model.input
    ## This model must already use linear activation for the final layer
    loss                 = model.layers[layer_idx].output[...,class_idx]
    grad_wrt_fmap        = K.gradients(loss,penultimate_output)[0]

    ## create function that evaluate the gradient for a given input
    # This function accept numpy array
    grad_wrt_fmap_fn     = K.function([layer_input,K.learning_phase()],
                                      [penultimate_output,grad_wrt_fmap])

    ## evaluate the derivative_fn
    fmap_eval, grad_wrt_fmap_eval = grad_wrt_fmap_fn([img[np.newaxis,...],0])

    # For numerical stability. Very small grad values along with small penultimate_output_value can cause
    # w * penultimate_output_value to zero out, even for reasonable fp precision of float32.
    grad_wrt_fmap_eval /= (np.max(grad_wrt_fmap_eval) + K.epsilon())

    print(grad_wrt_fmap_eval.shape)
    alpha_k_c           = grad_wrt_fmap_eval.mean(axis=(0,1,2)).reshape((1,1,1,-1))
    Lc_Grad_CAM         = np.maximum(np.sum(fmap_eval*alpha_k_c,axis=-1),0).squeeze()


class SFCorrCoeff(tf.keras.metrics.Metric):
    def __init__(self, name='stateful_corr_coeff', **kwargs):
        super(SFCorrCoeff, self).__init__(name=name, **kwargs)
        self.corr_coeff = self.add_weight(name='corr_coeff', initializer='zeros')
        #self.y_true_all = self.add_weight(name='y_true_all', initializer='zeros')
        #self.y_pred_all = self.add_weight(name='y_pred_all', initializer='zeros')
        self.y_true_all = []
        self.y_pred_all = []


    def update_state(self, y_true, y_pred, sample_weight=None):
        #self.y_true_all = K.concatenate((self.y_true_all, y_true))
        #self.y_pred_all = K.concatenate((self.y_pred_all, y_pred))
        self.y_true_all.append(y_true)
        self.y_pred_all.append(y_pred)

    def result(self):
        self.y_true_all = K.cast(self.y_true_all, K.floatx())
        self.y_pred_all = K.cast(self.y_pred_all, K.floatx())
        self.corr_coeff = tfp.stats.correlation(self.y_true_all, self.y_pred_all)
        return self.corr_coeff

    def reset_states(self):
        self.corr_coeff.assign(0)
        self.y_true_all.assign(None)
        self.y_pred_all.assign(None)
