# if K.backend() == 'theano':
#     from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
# if K.backend() == 'tensorflow':
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,Layer,
    BatchNormalization,
    Activation, Dense, Dropout
)
import numpy as np
import tensorflow.keras.backend as K
def theano_multinomial(n, pvals, seed):
    rng = RandomStreams(seed)
    return rng.multinomial(n=n, pvals=pvals, dtype='float32')

def tensorflow_categorical(count, seed):
    assert count > 0
    arr = [1.] + [.0 for _ in range(count-1)]
    return tf.random.shuffle(arr, seed)

# Returns a random array [x0, x1, ...xn] where one is 1 and the others
# are 0. Ex: [0, 0, 1, 0].
def rand_one_in_array(count, seed=False):
    if seed is False:
        seed = np.random.randint(1, 10e6)
    if K.backend() == 'theano':
        pvals = np.array([[1. / count for _ in range(count)]], dtype='float32')
        return theano_multinomial(n=1, pvals=pvals, seed=seed)[0]
    elif K.backend() == 'tensorflow':
        return tensorflow_categorical(count=count, seed=seed)
    else:
        raise Exception('Backend: {} not implemented'.format(K.backend()))

class JoinLayer(Layer):
    '''
    This layer will behave as Merge(mode='ave') during testing but
    during training it will randomly select between using local or
    global droppath and apply the average of the paths alive after
    aplying the drops.

    - Global: use the random shared tensor to select the paths.
    - Local: sample a random tensor to select the paths.
    '''

    def __init__(self, drop_p, is_global, global_path, force_path, **kwargs):
        #print "init"
        self.p = 1. - drop_p
        self.is_global = is_global
        self.global_path = global_path
        self.uses_learning_phase = True
        self.force_path = force_path
        super(JoinLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        #print("build")
        self.average_shape = list(input_shape[0])[1:]

    def _random_arr(self, count, p):
        return K.random_binomial((count,), p=p)

    def _arr_with_one(self, count):
        return rand_one_in_array(count=count)

    def _gen_local_drops(self, count, p):
        # Create a local droppath with at least one path
        arr = self._random_arr(count, p)
        drops = K.switch(
            K.any(arr),
            arr,
            self._arr_with_one(count)
        )
        return drops

    def _gen_global_path(self, count):
        return self.global_path[:count]

    def _drop_path(self, inputs):
        count = len(inputs)
        drops = K.switch(
            self.is_global,
            self._gen_global_path(count),
            self._gen_local_drops(count, self.p)
        )
        ave = K.zeros(shape=self.average_shape)
        for i in range(0, count):
            ave = inputs[i] * drops[i] + ave # += operation
        sum = K.sum(drops)
        # Check that the sum is not 0 (global droppath can make it
        # 0) to avoid divByZero
        ave = K.switch(
            K.not_equal(sum, 0.),
            ave/sum,
            ave)
        return ave

    def _ave(self, inputs):
        ave = inputs[0]
        for input in inputs[1:]:
            ave += input
        ave /= len(inputs)
        return ave

    def call(self, inputs, mask=False):
        #print("call")
        if self.force_path:
            output = self._drop_path(inputs)
        else:
            output = K.in_train_phase(self._drop_path(inputs), self._ave(inputs))
        return output

    def get_output_shape_for(self, input_shape):
        #print("get_output_shape_for", input_shape)
        return input_shape[0]

class JoinLayerGen:
    '''
    JoinLayerGen will initialize seeds for both global droppath
    switch and global droppout path.

    These seeds will be used to create the random tensors that the
    children layers will use to know if they must use global droppout
    and which path to take in case it is.
    '''

    def __init__(self, width, global_p=0.5, deepest=False):
        self.global_p = global_p
        self.width = width
        self.switch_seed = np.random.randint(1, 10e6)
        self.path_seed = np.random.randint(1, 10e6)
        self.deepest = deepest
        if deepest:
            self.is_global = K.variable(1.)
            self.path_array = K.variable([1.] + [.0 for _ in range(width-1)])
        else:
            self.is_global = self._build_global_switch()
            self.path_array = self._build_global_path_arr()

    def _build_global_path_arr(self):
        # The path the block will take when using global droppath
        return rand_one_in_array(seed=self.path_seed, count=self.width)

    def _build_global_switch(self):
        # A randomly sampled tensor that will signal if the batch
        # should use global or local droppath
        return K.equal(K.random_binomial((), p=self.global_p, seed=self.switch_seed), 1.)

    def get_join_layer(self, drop_p):
        global_switch = self.is_global
        global_path = self.path_array
        return JoinLayer(drop_p=drop_p, is_global=global_switch, global_path=global_path, force_path=self.deepest)

def fractal_base(self,input_len,output_len,last,dropout=False):
    self.input_len = input_len
    self.output_len = output_len
    def f(inputs):
        # print('inputs')
        # print(inputs)
        # print('inputs[0]')
        # print(inputs[0])
        # input('')

        # print('input_len')
        # print(self.input_len)
        # print('inputs')
        # print(inputs)
        # print('output_len')
        # print(output_len)
        # input('')
        # 1 hidden layer Dense is the base block
        
        if self.output_len == 8 and last == True:
            inputs = Dense(int(self.output_len),input_shape=(int(self.input_len),))(inputs)
        else:
            inputs = Dense(int(self.output_len),input_shape=(int(self.input_len),),activation='tanh')(inputs)
            # print('inputs after activation')
            # print(inputs)
            # input('')
        # print('inputs at 182')
        # print(inputs)
        # input('')
        # Unsure why inputs is [tensor] but needs to be removed with [0]
        # print('inputs[0]')
        # print(inputs[0])
        # input('')
        if dropout:
            inputs = Dropout(dropout)(inputs) # Removes nodes randomly
        # inputs = BatchNormalization(axis=-1 if K.backend() == 'theano' else -1,weights=None)(inputs)
            # print('inputs at 192')
            # print(inputs)
            # input('')
        
        return inputs
    return f

# XXX_ It's not clear when to apply Dropout, the paper cited
# (arXiv:1511.07289) uses it in the last layer of each stack but in
# the code gustav published it is in each convolution block so I'm
# copying it.
def fractal_block(self,join_gen,block,c,layersizes,i,drop_p, dropout=False):
    # print('Reached fractal_block')
    # input('')
    self.layersizes = layersizes
    self.c = c
    self.i = i
    def f(z):
        input_len = self.layersizes[self.i]
        # Function works by starting with the top-most row (reading from
        # left-most column as seen in model.png) per each block and merges once it hits
        # a Join row as seen in the paper. In the cifar10 case, c = 3
        # b = 5
        # z is the states input. c is the fractal width
        # Everytime function is called, it is starting on a new fractal block
        
        last = False
        output_len = self.layersizes[self.i]
        # print('input')
        # print(z)
        # input('')
        columns = z
        columns = [[z] for _ in range(c)]
        # print('columns')
        # print(columns[0])
        # input('')
        last_row = 2**(self.c-1) - 1 # last_row = 3; 4 rows on largest column
        for row in range(2**(self.c-1)): # row = (0,1,2,3)
            # print('row')
            # print(row)
            # input('')
            t_row = []
            for col in range(self.c): # executes 3 times (once for each column col=0,1,2). 0->1,1->2,2->4
                # print('col')
                # print(col)
                # input('')
                
                prop = 2**(col) # prop = 1,2,4
                # Add blocks
                if (row+1) % prop == 0: # executes 6 times row,prop = (0,1),(0,2),(0,4),(1,2),(1,4),(4,4)
                    # print('input_len')
                    # print(output_len)
                    # input('')
                    
                    t_col = columns[col] # col is 1,2,4, 2,4, 4 aka 0,1,2, 1,2, 2 t_col is [(32,32,3)]
                    # print('output_len')
                    # print(output_len)
                    # input('')
                    # print('t_col old')
                    # print(t_col)
                    # input('')    
                    t_col.append(fractal_base(self,input_len=input_len,
                                              output_len=output_len,
                                              last=last,
                                              dropout=dropout)(t_col[-1])) 
                    
                    if row == 3 and col == 2 and output_len == 32: # NEEDS TO BE CHANGED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        output_len = self.layersizes[self.i+1]

                    if row == 3 and col == 1 and output_len == 8:
                        last = True
                    
                    t_row.append(col) # col is 0,1, or 2 for c = 3
                    
                    # print('t_col new')
                    # print(t_col)
                    # input('')
                    # If last col appended is not 4, we need (layersizes[row+1]-layersizes[row])/4+previous
                # t_row; [0, 1], [0, 1, 2], [0, 1], [0, 1, 2]... happens 5 times
            # Merge ()
            if len(t_row) > 1: # only t_row = 0,1,2 and 0,1 executed
                # print("merging executed")
                merging = [columns[x][-1] for x in t_row] # identifying which conv sizes are needed to join together
                # print('merging')
                # print(merging)
                # input('')

                merged  = join_gen.get_join_layer(drop_p=drop_p)(merging)
                for i in t_row:
                    columns[i].append(merged)
            #     print('columns')
            #     print(columns)
            #     input('')
            # print('columns')
            # print(columns)
            # input('')
            # print('columns[0][-1]')
            # print(columns[0][-1])
            # input('')
        return columns[0][-1] # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Needs to be changed because output=z is no longer a set of 1x3 arrays 
        # takes the left most column's (the one with 4 sections) most recent addition
    # print('f returned in fractal_block')
    # print(f)
    # input('')
    return f# 

def fractal_net(self, bl, c, layersizes, drop_path, global_p=0.5, dropout=False, deepest=False):
    '''
    Return a function that builds the Fractal part of the network
    respecting keras functional model.
    When deepest is set, we build the entire network but set droppath
    to global and the Join masks to [1., 0... 0.] so only the deepest
    column is always taken.
    We don't add the softmax layer here nor build the model.
    '''
    # STILL NOT SURE WHY np NOT RECOGNIZED HEREs
    self.layersizes = layersizes
    def f(z):
        # !!!!!!!!!!!!!!!!!!!! This input, (z) will be changed from a 3D input to a vector input that will take in the observations !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        output = z
        # Initialize a JoinLayerGen that will be used to derive the
        # JoinLayers that share the same global droppath
        join_gen = JoinLayerGen(width=c, global_p=global_p, deepest=deepest) # c = 3, three paths per block
        for i in range(bl): # number of fractal blocks is bl (5)
            
                # -> arr_length, the array lengths,  described below !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # arr_length = number of nodes starting with 32, 29 observations as input + 3 dummy nodes. (e.g. 32,512,128,32,8)
                # -> frac_paths, number to be added 3 times to obtain next layer's # of nodes!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            dropout_i = dropout[i] if dropout else False # stores dropout probability or False if dropout is empty
            # print('output')
            # print(output)
            # input('')
            output = fractal_block(self,join_gen=join_gen,
                                   block=i,c=c, layersizes=self.layersizes,
                                   i=i,
                                   drop_p=drop_path,
                                   dropout=dropout_i)(output) # output = (depth,# rows, # cols)
            # I'm pretty sure maxpool2d not needed
            # print('output after fracatal block')
            # print(output)
            # input('')
        return output
    return f
