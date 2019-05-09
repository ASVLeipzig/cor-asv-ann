# -*- coding: utf-8
import unicodedata
import math
import logging
import pickle
import numpy as np
import h5py

from .alignment import Alignment

class Sequence2Sequence(object):
    '''Sequence to sequence (character-level) error correction with Keras.

    Adapted from examples/lstm_seq2seq.py (tutorial by Francois Chollet
    "A ten-minute introduction...") with changes as follows:

    - use early stopping to prevent overfitting
    - use all data, sorted into increments by increasing window length
    - window-based processing (with character alignment of training data)
    - use Dense instead of Embedding to allow input other than indexes
      (unit vectors): confidence and alternatives
    - use weight tying for output projection
    - add underspecification to character projection by conditioning
      index zero to lie in the center of other character vectors and
      randomly degrading input characters to zero during training
    - measure results not only on training set, but validation set as well
    - extend for use of large datasets: training uses generators on files
      with same generator function called twice (training vs validation),
      splitting lines via shared random variable
    - efficient preprocessing
    - use true zero for encoder padding and decoder start-of-sequence,
      use tab character for decoder padding (learned/not masked in training,
      treated like end-of-sequence in inference)
    - add runtime preprocessing function for convenient single-line testing
    - change first layer to bidirectional, stack unidirectional LSTM layers
      on top (with HL depth, HL width and window length configurable)
    - add beam search decoding (A*)
    - detect CPU vs GPU mode automatically
    - save/load weights separate from configuration (by recompiling model)
      in order to share weights between CPU and GPU model,
      and between fixed and variable batchsize/length
    - evaluate word and character error rates on separate dataset
    - add topology variant: stateful encoder mode
    - add topology variant: deep bi-directional encoder
    - add topology variant: residual connections
    - add topology variant: dense bridging transfer
    - add training variant: scheduled sampling
    - add training variant: parallel LM loss
    - allow incremental training (e.g. pretraining on clean text)
    - allow weight transfer from shallower model (fixing shallow layer
      weights) or from language model (as unconditional decoder),
      update character mapping and re-use character embeddings
    - allow resetting encoder after load/init transfer
    
    Features still (very much) wanting of implementation:
    
    - attention or at least peeking (not just final-initial transfer)
    - stateful decoder mode (in non-transfer part of state function)
    - systematic hyperparameter treatment (deviations from Sutskever
      should be founded on empirical analysis):
      HL width and depth, optimiser choice (RMSprop/SGD) and parameters,
      gradient clipping, decay and rate control, initialisers
    
    # Summary of the algorithm
    
    - We start with source sequences from a domain (e.g. English sentences)
        and correspding target sequences from another domain
        (e.g. French sentences).
    - An encoder LSTM turns source sequences to 2 state vectors
        (we keep the last LSTM state and discard the outputs).
    - A decoder LSTM is trained to turn the target sequences into
        the same sequence but offset by one timestep in the future,
        a training process called "teacher forcing" in this context.
        Is uses as initial state the state vectors from the encoder.
        Effectively, the decoder learns to generate `targets[t+1...]`
        given `targets[...t]`, conditioned on the source sequence.
    - In inference mode, when we want to decode unknown source sequences, we:
        - Encode the source sequence into state vectors
        - Start with a target sequence of size 1
            (just the start-of-sequence character)
        - Feed the state vectors and 1-char target sequence
            to the decoder to produce predictions for the next character
        - Sample the next character using these predictions
            (we simply use argmax).
        - Append the sampled character to the target sequence
        - Repeat until we generate the end-of-sequence character or we
            hit the character limit.
    
    # References
    
    - Sequence to Sequence Learning with Neural Networks
        https://arxiv.org/abs/1409.3215
    - Learning Phrase Representations using
        RNN Encoder-Decoder for Statistical Machine Translation
        https://arxiv.org/abs/1406.1078
    '''

    def __init__(self, logger=None, progbars=True):
        # model parameters
        self.batch_size = 64  # How many samples are trained together? (batch length)
        self.window_length = 7 # How many bytes are encoded at once? (sequence length)
        self.stateful = False # stateful encoder (implicit state transfer between batches)
        self.width = 512  # latent dimensionality of the encoding space (hidden layer state)
        self.depth = 2 # number of encoder/decoder layers stacked above each other (only 1st layer will be BLSTM)
        self.mapping = ({'': 0}, {0: ''}) # indexation of (known/allowed) input and output characters (i.e. vocabulary)
        self.voc_size = 1 # size of mapping (0 reserved for padding)
        # note: character mapping includes nul for non-output, tab for start, newline for stop;
        #       mapping/voc_size set by loading or training
        self.residual_connections = False # add input to output in encoder and decoder HL
        self.deep_bidirectional_encoder = False # encoder HL are all BLSTM cross-summarizing forward and backward outputs (as -encoder_type bdrnn in Open-NMT)
        self.bridge_dense = False # use a FFNN to map encoder final states to decoder initial states instead of copy
        
        # training parameters
        self.epochs = 100  # maximum number of epochs to train for (unless stopping early by validation loss)
        self.lm_loss = False # train with additional output (unweighted sum loss) from LM, defined with tied decoder weights and same input but not conditioned on encoder output (applies to encoder_decoder_model only, does not affect encoder_model and decoder_model)
        self.scheduled_sampling = None # 'linear'/'sigmoid'/'exponential'/None # train with additional output from self-loop (softmax feedback) instead of teacher forcing (with loss weighted by given curve across epochs), defined with tied weights and same encoder output (applies to encoder_decoder_model only, does not affect encoder_model and decoder_model)
        self.dropout = 0.2 # rate of dropped input connections in encoder and decoder HL during training
        
        # inference parameters
        self.beam_width_in = 4 # up to how many new candidates can enter the beam in decode_sequence_beam()?
        self.beam_width_out = 10 # up to how many results can be drawn from generator decode_sequence_beam()?

        self.logger = logger or logging.getLogger(__name__)
        self.graph = None # for tf access from multiple threads
        self.encoder_decoder_model = None # combined model for training
        self.encoder_model = None # separate model for inference
        self.decoder_model = None # separate model for inference
        self.aligner = Alignment(0) # aligner (for windowing and/or evaluation) with internal state

        self.progbars = progbars
        self.status = 0 # empty / configured / trained?
    
    def configure(self, batch_size=None):
        '''Define and compile encoder and decoder models for the configured parameters.
        
        Use given `batch_size` for encoder input if stateful:
        configure once for training phase (with parallel lines),
        then reconfigure for prediction (with only 1 line each).
        (Decoder input will always have `self.batch_size`,
        either from parallel input lines during training phase,
        or from parallel hypotheses during prediction.)
        '''
        from keras.initializers import RandomNormal
        from keras.layers import Input, Dense, TimeDistributed, Dropout
        from keras.layers import LSTM, CuDNNLSTM, Bidirectional, Lambda
        from keras.layers import concatenate, average, add
        from keras.models import Model
        #from keras.utils import plot_model
        from keras import backend as K
        import tensorflow as tf
        
        if batch_size:
            self.batch_size = batch_size
        
        # self.sess = tf.Session()
        # K.set_session(self.sess)
        
        # automatically switch to CuDNNLSTM if CUDA GPU is available:
        has_cuda = K.backend() == 'tensorflow' and K.tensorflow_backend._get_available_gpus()
        self.logger.info('using %s LSTM implementation to compile %s model '
                         'of depth %d width %d window length %d size %d',
                         'GPU' if has_cuda else 'CPU',
                         'stateful' if self.stateful else 'stateless',
                         self.depth, self.width, self.window_length, self.voc_size)
        if self.residual_connections:
            self.logger.info('encoder and decoder LSTM outputs are added to inputs in all hidden layers'
                             '(residual_connections)')
        if self.deep_bidirectional_encoder:
            self.logger.info('encoder LSTM is bidirectional in all hidden layers, '
                             'with fw/bw cross-summation between layers (deep_bidirectional_encoder)')
        if self.bridge_dense:
            self.logger.info('state transfer between encoder and decoder LSTM uses '
                             'non-linear Dense layer as bridge in all hidden layers (bridge_dense)')
        lstm = CuDNNLSTM if has_cuda else LSTM
        
        ### Define training phase model
        
        # Set up an input sequence and process it.
        if self.stateful:
            # batch_size = 1 # override does not work (re-configuration would break internal encoder updates)
            encoder_inputs = Input(batch_shape=(self.batch_size, self.window_length, self.voc_size),
                                   name='encoder_input')
        else:
            encoder_inputs = Input(shape=(self.window_length, self.voc_size),
                                   name='encoder_input')
        char_embedding = Dense(self.width, use_bias=False,
                               kernel_initializer=RandomNormal(stddev=0.001),
                               kernel_regularizer=self._regularise_chars,
                               name='char_embedding')
        encoder_outputs = char_embedding(encoder_inputs)
        
        if self.deep_bidirectional_encoder:
            # cross-summary here means: i_next_fw[k] = i_next_bw[k] = o_fw[k-1]+o_bw[k-1]
            # i.e. add flipped fw/bw outputs by reshaping last axis into half-width axis and 2-dim axis,
            # then reversing the last and reshaping back;
            # in numpy this would be:
            #     x + np.flip(x.reshape(x.shape[:-1] + (int(x.shape[-1]/2),2)), -1).reshape(x.shape))
            # in keras this would be something like this (but reshape requires TensorShape no list/tuple):
            #     x + K.reshape(K.reverse(K.reshape(x, K.int_shape(x)[:-1] + (x.shape[-1].value//2,2)), axes=-1), x.shape)
            # in tensorflow this would be (but does not work with batch_shape None):
            #     x + tf.reshape(tf.reverse(tf.reshape(x, tf.TensorShape(x.shape.as_list()[:-1] + [x.shape[-1].value//2, 2])), [-1]), x.shape)
            # it finally works by replacing all None dimensions with -1:
            cross_sum = Lambda(lambda x: x + tf.reshape(
                tf.reverse(tf.reshape(x, [-1, x.shape[1].value, x.shape[2].value//2, 2]), [-1]),
                [-1] + x.shape.as_list()[1:]))
            # TODO: pyramidal cross-summary means also multiplexing timesteps:
            # i_next_fw[k,t] = i_next_bw[k,t] = o_fw[k-1,t]+o_bw[k-1,t]+o_fw[k-1,t-1]+o_bw[k-1,t-1]
        
        encoder_state_outputs = []
        for n in range(self.depth):
            args = {'name': 'encoder_lstm_%d' % (n+1),
                    'return_state': True,
                    'return_sequences': (n < self.depth-1),
                    'stateful': self.stateful}
            if not has_cuda:
                # instead of default 'hard_sigmoid' which deviates from CuDNNLSTM:
                args['recurrent_activation'] = 'sigmoid'
            layer = lstm(self.width, **args)
            if n == 0 or self.deep_bidirectional_encoder:
                encoder_outputs, fw_state_h, fw_state_c, bw_state_h, bw_state_c = (
                    Bidirectional(layer, name=layer.name)(
                        encoder_outputs if n == 0 else cross_sum(encoder_outputs)))
                # prepare for current layer decoder initial_state:
                state_h = average([fw_state_h, bw_state_h]) # concatenate
                state_c = average([fw_state_c, bw_state_c]) # concatenate
            else:
                encoder_outputs2, state_h, state_c = layer(encoder_outputs)
                if self.residual_connections:
                    # add residual connections:
                    if n == 1:
                        #encoder_outputs = add([encoder_outputs2, average([encoder_outputs[:,:,::2], encoder_outputs[:,:,1::2]])]) # does not work (no _inbound_nodes)
                        encoder_outputs = encoder_outputs2
                    else:
                        encoder_outputs = add([encoder_outputs2, encoder_outputs])
                else:
                    encoder_outputs = encoder_outputs2
            if n > 0: # only hidden-to-hidden/output layer:
                if self.stateful:
                    constant_shape = (self.batch_size, 1, self.width) # variational dropout (time-constant)
                else:
                    constant_shape = (1, self.width) # variational dropout (time-constant)
                # LSTM (but not CuDNNLSTM) has the (non-recurrent) dropout keyword option for this:
                encoder_outputs = Dropout(self.dropout, noise_shape=constant_shape)(encoder_outputs)
            if self.bridge_dense:
                state_h = Dense(self.width, activation='tanh', name='bridge_state_h')(state_h)
                state_c = Dense(self.width, activation='tanh', name='bridge_state_c')(state_c)
            encoder_state_outputs.extend([state_h, state_c])
        
        # Set up an input sequence and process it.
        if self.stateful:
            # shape inference would assume fixed batch size here as well
            # (but we need that to be flexible for prediction):
            decoder_inputs = Input(batch_shape=(None, None, self.voc_size),
                                   name='decoder_input')
        else:
            decoder_inputs = Input(shape=(None, self.voc_size),
                                   name='decoder_input')
        decoder_outputs0 = char_embedding(decoder_inputs)
        decoder_outputs = decoder_outputs0
        
        # Set up decoder to return full output sequences (so we can train in parallel),
        # to use encoder_state_outputs as initial state and return final states as well.
        # We don't use those states in the training model, but we will use them
        # for inference (see further below).
        decoder_lstms = []
        for n in range(self.depth):
            args = {'name': 'decoder_lstm_%d' % (n+1),
                    'return_state': True,
                    'return_sequences': True}
            if not has_cuda:
                # instead of default 'hard_sigmoid' which deviates from CuDNNLSTM:
                args['recurrent_activation'] = 'sigmoid'
            layer = lstm(self.width, **args) # self.width*2 if n == 0 else
            decoder_lstms.append(layer)
            decoder_outputs2, _, _ = layer(decoder_outputs,
                                           initial_state=encoder_state_outputs[2*n:2*n+2])
            # add residual connections:
            if n > 0 and self.residual_connections:
                decoder_outputs = add([decoder_outputs2, decoder_outputs])
            else:
                decoder_outputs = decoder_outputs2
            decoder_outputs = Dropout(self.dropout)(decoder_outputs)
        
        def decoder_output(x):
            # re-use input embedding (weight tying), but add a bias vector,
            # and also add a linear projection in hidden space
            # (see Press & Wolf 2017)
            # y = softmax( V * P * h + b ) with V=U the input embedding;
            # initialise P as identity matrix and b as zero
            #proj = K.variable(np.eye(self.width), name='char_output_projection') # trainable=True by default
            #bias = K.variable(np.zeros((self.voc_size,)), name='char_output_bias') # trainable=True by default
            #return K.softmax(K.dot(h, K.transpose(K.dot(char_embedding.embeddings, proj))) + bias)
            # simplified variant with no extra weights (50% faster, equally accurate):
            return K.softmax(K.dot(x, K.transpose(char_embedding.kernel)))
        # for experimenting with global normalization in beam search
        # (gets worse if done just like that): 'sigmoid'
        char_projection = TimeDistributed(Lambda(decoder_output, name='transpose+softmax'),
                                          name='char_projection')
        decoder_outputs = char_projection(decoder_outputs)
        
        if self.lm_loss:
            lm_outputs = decoder_outputs0
            for n in range(self.depth):
                layer = decoder_lstms[n] # tied weights
                lm_outputs, _, _ = layer(lm_outputs)
            lm_outputs = char_projection(lm_outputs)
            
            decoder_outputs = [decoder_outputs, lm_outputs] # 2 outputs, 1 combined loss
        
        # Bundle the model that will turn
        # `encoder_input_data` and `decoder_input_data` into `decoder_output_data`
        self.encoder_decoder_model = Model([encoder_inputs, decoder_inputs], decoder_outputs,
                                           name='encoder_decoder_model')
        
        ## Define inference phase model
        # Here's the drill:
        # 1) encode source to retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        #    and a "start of sequence" as target token.
        # 3) repeat from 2, feeding back the target token
        #    from output to input, and passing state
        
        # Re-use the training phase encoder unchanged
        # (with result states as output).
        self.encoder_model = Model(encoder_inputs, encoder_state_outputs,
                                   name='encoder_model')
        
        # Set up decoder differently: with additional input
        # as initial state (not just encoder_state_outputs), and
        # keeping final states (instead of discarding), 
        # so we can pass states explicitly.
        decoder_state_inputs = []
        decoder_state_outputs = []
        decoder_outputs = decoder_outputs0
        for n in range(self.depth):
            state_h_in = Input(shape=(self.width,), # self.width*2 if n == 0 else
                               name='initial_h_%d_input' % (n+1))
            state_c_in = Input(shape=(self.width,), # self.width*2 if n == 0 else
                               name='initial_c_%d_input' % (n+1))
            decoder_state_inputs.extend([state_h_in, state_c_in])
            layer = decoder_lstms[n]
            decoder_outputs, state_h_out, state_c_out = layer(
                decoder_outputs,
                initial_state=decoder_state_inputs[2*n:2*n+2])
            decoder_state_outputs.extend([state_h_out, state_c_out])
        decoder_outputs = char_projection(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_state_inputs,
            [decoder_outputs] + decoder_state_outputs,
            name='decoder_model')
        
        ## Compile model
        self.recompile()
        # for tf access from multiple threads
        # self.encoder_model._make_predict_function()
        # self.decoder_model._make_predict_function()
        # self.sess.run(tf.global_variables_initializer())
        self.graph = tf.get_default_graph()
        self.status = 1
    
    def recompile(self):
        from keras.optimizers import Adam
        
        self.encoder_decoder_model.compile(
            loss='categorical_crossentropy', # loss_weights=[1.,1.] if self.lm_loss
            optimizer=Adam(clipnorm=5), #'adam',
            sample_weight_mode='temporal') # sample_weight slows down training slightly (20%)
    
    def reconfigure_for_mapping(self):
        '''Reconfigure character embedding layer after change of mapping (possibly transferring previous weights).'''
        
        assert self.status >= 1
        embedding = self.encoder_decoder_model.get_layer(name='char_embedding')
        input_dim = embedding.input_spec.axes[-1]
        if input_dim < self.voc_size: # more chars than during last training?
            if self.status >= 2: # weights exist already (i.e. incremental training)?
                self.logger.warning('transferring weights from previous model with only %d character types', input_dim)
                # get old weights:
                layer_weights = [layer.get_weights() for layer in self.encoder_decoder_model.layers]
                # reconfigure with new mapping size (and new initializers):
                self.configure()
                # set old weights:
                for layer, weights in zip(self.encoder_decoder_model.layers, layer_weights):
                    self.logger.debug('transferring weights for layer %s %s', layer.name, str([w.shape for w in weights]))
                    if layer.name == 'char_embedding':
                        # transfer weights from previous Embedding layer to new one:
                        new_weights = layer.get_weights() # freshly initialised
                        #new_weights[0][input_dim:, 0:embedding.units] = weights[0][0,:] # repeat zero vector instead
                        new_weights[0][0:input_dim, 0:embedding.units] = weights[0]
                        layer.set_weights(new_weights)
                    else:
                        # use old weights:
                        layer.set_weights(weights)
            else:
                self.configure()
    
    def _regularise_chars(self, embedding_matrix):
        '''Calculate L2 loss of the char embedding weights
        to control for underspecification at zero
        (by interpolating between other embedding vectors).
        '''
        from keras import backend as K
        
        em_dims = embedding_matrix.shape.as_list()
        if em_dims[0] == 0: # voc_size starts with 0 before first training
            return 0
        vec0 = K.slice(embedding_matrix, [0, 0], [1, em_dims[1]])            # zero vector only,
        #vec0 = K.repeat_elements(vec0, em_dims[0]-1, axis=0)                 # repeated
        vecs = K.slice(embedding_matrix, [1, 0], [em_dims[0]-1, em_dims[1]]) # all vectors except zero
        # make sure only vec0 is affected, i.e. vecs change only via global loss:
        vecs = K.stop_gradient(K.mean(vecs, axis=0))
        # scale to make gradients benign:
        underspecification = 1 * K.sum(K.square(vec0 - vecs)) # c='\0' ~ mean of others

        #lowrank = K.sum(0.01 * K.square(embedding_matrix)) # generalization/sparsity
        norms = K.sum(K.square(embedding_matrix), axis=1)
        norm0 = K.ones_like(norms) # square of target (non-zero) weight norm
        lowrank = 0.01 * K.sum(K.square(norm0 - norms))
        
        return lowrank + underspecification
    
    def train(self, filenames):
        '''train model on text files
        
        Pass the character sequence of lines in `filenames`,
        paired into source and target and aligned into fixed-length windows,
        to the loop training model weights with stochastic gradient descent.
        The generator will open the file, looping over the complete set (epoch)
        as long as validation error does not increase in between (early stopping).
        Validate on a random fraction of lines automatically separated before.
        (Data are always split by line, regardless of stateless/stateful mode.)
        '''
        from keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN
        from .callbacks import StopSignalCallback, ResetStatesCallback
        from .keras_train import fit_generator_autosized, evaluate_generator_autosized
        
        # Run training
        earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1,
                                      mode='min', restore_best_weights=True)
        callbacks = [earlystopping, TerminateOnNaN(),
                     StopSignalCallback(logger=self.logger)]
        if self.stateful: # reset states between batches of different lines/documents (controlled by generator)
            reset_cb = ResetStatesCallback(self.encoder_model, logger=self.logger)
            callbacks.append(reset_cb)
        else:
            reset_cb = None
            
        # todo: shuffle lines
        num_lines = 0
        chars = set(self.mapping[0].keys()) # includes '' (0)
        for filename in filenames:
            # todo: there must be a better way to detect this:
            with_confidence = filename.endswith('.pkl')
            with open(filename, 'rb' if with_confidence else 'r') as file:
                if with_confidence:
                    file = pickle.load(file) # read once
                for line in file:
                    if with_confidence:
                        source_conf, target_text = line
                        line = ''.join([char for char, prob in source_conf]) + '\t' + target_text
                    line = unicodedata.normalize('NFC', line)
                    chars.update(set(line))
                    num_lines += 1
        chars = sorted(list(chars))
        if len(chars) > self.voc_size:
            # incremental training
            c_i = dict((c, i) for i, c in enumerate(chars))
            i_c = dict((i, c) for i, c in enumerate(chars))
            self.mapping = (c_i, i_c)
            self.voc_size = len(c_i)
            self.reconfigure_for_mapping()
        self.logger.info('Training on "%d" files with %d lines', len(filenames), num_lines)
        split_rand = np.random.uniform(0, 1, (num_lines,)) # reserve split fraction at random line numbers
        
        history = fit_generator_autosized(
            self.encoder_decoder_model,
            self.gen_data(filenames, split_rand, train=True, reset_cb=reset_cb),
            epochs=self.epochs,
            workers=1,
            # (more than 1 would effectively increase epoch size)
            use_multiprocessing=not self.scheduled_sampling and not self.stateful,
            # (cannot access session/graph for scheduled sampling in other process,
            #  cannot access model for reset callback in other process)
            validation_data=self.gen_data(filenames, split_rand, train=False, reset_cb=reset_cb),
            verbose=1 if self.progbars else 0,
            callbacks=callbacks,
            validation_callbacks=[reset_cb] if self.stateful else None)
        
        if 'val_loss' in history.history:
            self.logger.info('training finished with val_loss %f',
                             min(history.history['val_loss']))
            if np.isnan(history.history['loss'][-1]):
                # recover weights (which TerminateOnNaN prevented EarlyStopping from doing)
                self.encoder_decoder_model.set_weights(earlystopping.best_weights)
            self.status = 2
        else:
            self.logger.critical('training failed')
            self.status = 1
    
    def evaluate(self, filenames):
        '''evaluate model on text files
        
        Pass the character sequence of lines in `filenames`,
        paired into source and target and aligned into fixed-length windows,
        to a loop predicting outputs with decoder feedback and greedy+beam search.
        The generator will open the file, looping over the complete set once,
        printing source/target and predicted lines (recombining windows),
        and the overall calculated character and word error rates of source (OCR)
        and prediction (greedy/beamed) against target (GT).
        (Data are always split by line, regardless of stateless/stateful mode.)
        '''
        assert self.status == 2
        #would still be teacher forcing:
        # loss = s2s.encoder_decoder_model.evaluate_generator_autosized(s2s.gen_data(filename))
        # output = s2s.encoder_decoder_model.predict_generator_autosized(s2s.gen_data(filename))
        counts = [1e-9, 0, 0, 0, 1e-9, 0, 0, 0]
        for batch_no, batch in enumerate(self.gen_windows(filenames, False)):
            source_lines, target_lines, sourceconf_lines = batch
            #bar.update(1)
            if self.stateful:
                # model controlled by caller (batch prediction)
                #self.logger.debug('resetting encoder for line', line_no, train)
                # does not work for some reason (never returns,
                # even if passing self.sess.as_default() and self.graph.as_default()):
                #self.encoder_decoder_model.reset_states()
                self.encoder_model.reset_states()
                #self.decoder_model.reset_states() # not stateful yet
            
            source_texts = []
            target_texts = []
            greedy_texts = []
            beamed_texts = []
            for i in range(max([len(line) for line in source_lines])):
                # vectorize:
                encoder_input_data, decoder_input_data, _ = (
                    self.vectorize_windows(
                        [line[i] if len(line) > i else '' for line in source_lines],
                        [line[i] if len(line) > i else '' for line in target_lines],
                        [line[i] if len(line) > i else [] for line in sourceconf_lines] \
                        if sourceconf_lines else None))
                
                # avoid repeating encoder in both functions (greedy and beamed),
                # because we can only reset at the first window
                source_states = self.encoder_model.predict_on_batch(encoder_input_data)
                for j in range(self.batch_size):
                    if i == 0:
                        source_texts.append(u'')
                        target_texts.append(u'')
                        greedy_texts.append(u'')
                        beamed_texts.append(u'')
                    if i >= len(source_lines[j]) or i >= len(target_lines[j]):
                        continue # avoid empty window (masked during training)
                    # Take one sequence (part of the training/validation set) and try decoding it
                    source_texts[j] += source_lines[j][i]
                    target_texts[j] += target_lines[j][i]
                    greedy_texts[j] += self.decode_sequence_greedy(
                        source_state=[layer[j:j+1] for layer in source_states])
                    try: # query only 1-best
                        beamed_texts[j] += next(self.decode_sequence_beam(
                            source_state=[layer[j:j+1] for layer in source_states],
                            eol=(i + 1 >= len(source_lines[j]))))
                    except StopIteration:
                        self.logger.error('no beam decoder result within processing limits for '
                                          '"%s\t%s" window %d of %d',
                                          source_texts[j].rstrip(), target_texts[j].rstrip(),
                                          i+1, len(source_lines[j]))
            
            for j in range(self.batch_size):
                if not source_lines[j] or not target_lines[j]:
                    # ignore (zero) remainder of partially filled last batch
                    continue
                line_no = batch_no * self.batch_size + j
                
                self.logger.info('Source input              : %s', source_texts[j].rstrip(u'\n'))
                self.logger.info('Target output             : %s', target_texts[j].rstrip(u'\n'))
                self.logger.info('Target prediction (greedy): %s', greedy_texts[j].rstrip(u'\n'))
                self.logger.info('Target prediction (beamed): %s', beamed_texts[j].rstrip(u'\n'))
                
                #metric = self.aligner.get_levenshtein_distance
                metric = self.aligner.get_adjusted_distance
                
                c_edits_ocr, c_total = metric(source_texts[j], target_texts[j])
                c_edits_greedy, _ = metric(greedy_texts[j], target_texts[j])
                c_edits_beamed, _ = metric(beamed_texts[j], target_texts[j])
                
                greedy_tokens = greedy_texts[j].split(" ")
                beamed_tokens = beamed_texts[j].split(" ")
                source_tokens = source_texts[j].split(" ")
                target_tokens = target_texts[j].split(" ")

                w_edits_ocr, w_total = metric(source_tokens, target_tokens)
                w_edits_greedy, _ = metric(greedy_tokens, target_tokens)
                w_edits_beamed, _ = metric(beamed_tokens, target_tokens)
                
                counts[0] += c_total
                counts[1] += c_edits_ocr
                counts[2] += c_edits_greedy
                counts[3] += c_edits_beamed
                counts[4] += w_total
                counts[5] += w_edits_ocr
                counts[6] += w_edits_greedy
                counts[7] += w_edits_beamed
        
        self.logger.info("CER OCR:    %.3f", counts[1] / counts[0])
        self.logger.info("CER greedy: %.3f", counts[2] / counts[0])
        self.logger.info("CER beamed: %.3f", counts[3] / counts[0])
        self.logger.info("WER OCR:    %.3f", counts[5] / counts[4])
        self.logger.info("WER greedy: %.3f", counts[6] / counts[4])
        self.logger.info("WER beamed: %.3f", counts[7] / counts[4])

    # for fit_generator()/predict_generator()/evaluate_generator()/standalone
    # -- looping, but not shuffling
    def gen_data(self, filenames, split=None, train=False, reset_cb=None):
        '''generate batches of vector data from text file
        
        Open `filenames` in text mode, loop over them producing one window
        of batch_size lines at a time.
        Pad windows to a `self.window_length` multiple of the longest line,
        respectively.
        If stateful, call `reset_cb` at the start of each line (if given)
        or resets model directly (otherwise).
        Skip lines at `split` positions (if given), depending on `train`
        (upper vs lower partition).
        Yields vector data batches (for fit_generator/evaluate_generator).
        '''
        import threading
        
        epoch = 0
        if train and self.scheduled_sampling:
            sample_ratio = 0
        lock = threading.Lock()
        for batch in self.gen_windows(filenames, True, split, train):
            if not batch:
                epoch += 1
                yield False # signal end of epoch to autosized fit/evaluate
                if train and self.scheduled_sampling:
                    # prepare empirical scheduled sampling (i.e. without proper gradient)
                    attenuation = 3 # 10 enters saturation at about 10 percent of self.epochs
                    if self.scheduled_sampling == 'linear':
                        sample_ratio = attenuation * (epoch - 1) / (self.epochs - 1)
                    elif self.scheduled_sampling == 'sigmoid':
                        sample_ratio = 1 / (1 + math.exp(5 - 10 * attenuation * epoch / self.epochs))
                    elif self.scheduled_sampling == 'exponential':
                        sample_ratio = 1 - 0.9 ** (50 * attenuation * epoch / self.epochs)
                    else:
                        raise Exception('unknown function "%s" for scheduled sampling' % self.scheduled_sampling)
                    #self.logger.debug('sample ratio for this epoch:', sample_ratio)
            else:
                source_lines, target_lines, sourceconf_lines = batch
                if train and self.scheduled_sampling:
                    line_schedules = np.random.uniform(0, 1, self.batch_size)
                else:
                    line_schedules = None
                if reset_cb and self.stateful:
                    # model controlled by callbacks (training)
                    reset_cb.reset("")
                lock.acquire() # ensure no other generator instance interferes within the block of lines
                for i in range(max([len(line) for line in source_lines])):
                    # vectorize:
                    encoder_input_data, decoder_input_data, decoder_output_data = (
                        self.vectorize_windows(
                            [line[i] if len(line) > i else '' for line in source_lines],
                            [line[i] if len(line) > i else '' for line in target_lines],
                            [line[i] if len(line) > i else [] for line in sourceconf_lines] \
                            if sourceconf_lines else None))
                    # yield source/target data to keras consumer loop (fit/evaluate)
                    if line_schedules is not None: # and epoch > 1:
                        # calculate greedy/beamed decoder output to yield as as decoder input
                        window_nonempty = np.array(list(map(lambda target, windows=i: len(target) > windows,
                                                            target_lines))) # avoid lines with empty windows
                        line_scheduled = line_schedules < sample_ratio # respect current schedule
                        indexes = np.logical_and(line_scheduled, window_nonempty)
                        if np.count_nonzero(indexes) > 0:
                            # ensure the generator thread gets to see the same tf graph:
                            # with self.sess.as_default():
                            with self.graph.as_default():
                                decoder_input_data_sampled = self.decode_batch_greedy(encoder_input_data)
                                # overwrite scheduled lines with data sampled from decoder instead of GT:
                                indexes_condition = np.broadcast_to(indexes, # broadcast to data shape
                                                                    tuple(reversed(decoder_input_data.shape))).transpose()
                                decoder_input_data = np.where(indexes_condition,
                                                              decoder_input_data_sampled, decoder_input_data)
                                #self.logger.debug('sampled %02d lines for window %d', np.count_nonzero(indexes), i)
                    if train:
                        # encoder degradation to index zero for learning character underspecification
                        rand = np.random.uniform(0, 1, self.batch_size)
                        rand = (self.window_length * rand / 0.01).astype(np.int) # effective degradation ratio
                        encoder_input_data[np.arange(self.batch_size)[rand < self.window_length],
                                           rand[rand < self.window_length], :] = np.eye(self.voc_size)[0]
                    # sample_weight quickly causes getting stuck with NaN,
                    # both in gradient updates and weights (regardless of
                    # loss function, optimizer, gradient clipping, CPU or GPU)
                    # when re-training, so disable
                    yield ([encoder_input_data, decoder_input_data], decoder_output_data)
                lock.release()
                    
    def gen_windows(self, filenames, repeat=True, split=None, train=False):
        split_ratio = 0.2
        epoch = 0
        while True:
            source_lines = []
            target_lines = []
            sourceconf_lines = []
            for filename in filenames:
                with_confidence = filename.endswith('.pkl')
                with open(filename, 'rb' if with_confidence else 'r') as file:
                    if with_confidence:
                        file = pickle.load(file) # read once
                    #if (repeat and not with_confidence):
                    #    file.seek(0) # read again
                    for line_no, line in enumerate(file):
                        if (isinstance(split, np.ndarray) and
                            (split[line_no] < split_ratio) == train):
                            # data shared between training and validation: belongs to other generator, resp.
                            #print('skipping line %d in favour of other generator' % line_no)
                            continue
                        if with_confidence: # binary input with OCR confidence?
                            source_conf, target_text = line # already includes end-of-sequence
                            source_text = u''.join([char for char, prob in source_conf])
                            # start-of-sequence will be added window by window,
                            # end-of-sequence already preserved by pickle format
                        else:
                            source_text, target_text = line.split('\t')
                            # add end-of-sequence:
                            source_text = source_text + u'\n'
                            # start-of-sequence will be added window by window,
                            # end-of-sequence already preserved by file iterator:
                            target_text = target_text
                        source_text = unicodedata.normalize('NFC', source_text)
                        target_text = unicodedata.normalize('NFC', target_text)

                        # align source and target text line, shelve them into successive fixed-size windows
                        self.aligner.set_seqs(source_text, target_text)
                        if train and self.aligner.is_bad():
                            if epoch == 0:
                                self.logger.debug('%s' 'ignoring bad line "%s\t%s"',
                                                  '\x1b[2K\x1b[G' if self.progbars else '',
                                                  source_text.rstrip(), target_text.rstrip())
                            continue # avoid training if OCR was too bad
                        alignment1 = self.aligner.get_best_alignment()
                        if with_confidence: # binary input with OCR confidence?
                            # multiplex confidences into string alignment result:
                            k = 0
                            for i, (source_char, target_char) in enumerate(alignment1):
                                conf = 0
                                if source_char != self.aligner.gap_element:
                                    assert source_char == source_conf[k][0], (
                                        "characters from alignment and from confidence tuples out of sync at " +
                                        "{} in \"{}\" \"{}\": {}".format(k, source_text, target_text, alignment1))
                                    conf = source_conf[k][1]
                                    k += 1
                                alignment1[i] = ((source_char, conf), target_char)

                        # Produce batches of training data:
                        # - fixed `self.batch_size` lines
                        #   (each line post-padded to maximum window multiple among batch)
                        # - fixed `self.window_length` samples
                        #   (each window post-padded to fixed sequence length)
                        # with decoder windows twice as long as encoder windows.
                        # Yield window by window when full, then inform callback `reset_cb`.
                        # Callback will do `reset_states()` on all stateful layers of model
                        # during `on_batch_begin`.
                        # If stateful, after arriving at the very end of the file with
                        # a partial batch, wrap around to complete it, randomly selecting
                        # the required number of lines again.
                        source_windows, target_windows, sourceconf_windows = (
                            self.window_line(alignment1,
                                             with_confidence=with_confidence,
                                             strict=train, verbose=(epoch == 0 and self.progbars)))
                        source_lines.append(source_windows)
                        target_lines.append(target_windows)
                        if with_confidence:
                            sourceconf_lines.append(sourceconf_windows)

                        if len(source_lines) == self.batch_size: # end of batch
                            yield (source_lines, target_lines,
                                   sourceconf_lines if with_confidence else None)
                            source_lines = []
                            target_lines = []
                            sourceconf_lines = []
            epoch += 1
            if repeat:
                yield False
                # bury remaining lines (partially filled batch)
            else:
                if source_lines:
                    # a partially filled batch remains
                    source_lines.extend((self.batch_size-len(source_lines))*[[]])
                    target_lines.extend((self.batch_size-len(source_lines))*[[]])
                    if with_confidence:
                        sourceconf_lines.extend((self.batch_size-len(source_lines))*[[]])
                    yield (source_lines, target_lines,
                           sourceconf_lines if with_confidence else None)
                break
    
    def window_line(self, line, with_confidence=False, strict=False, verbose=False):
        # Ensure that source/target window does not become empty
        # (avoid by moving last characters from previous window).
        source_windows = ['']
        target_windows = ['']
        sourceconf_windows = [[]]
        i = 0
        j = 0
        try:
            for source_char, target_char in line:
                if with_confidence: # binary input with OCR confidence?
                    source_char, source_conf = source_char
                if (i+1 > self.window_length -3 or
                    j+1 > self.window_length -3): # window already full?
                    # or source_char == u' ' or target_char == u' '
                    if (strict and i == 0 and
                        target_windows[-1].strip(u'—-. ')):
                        # empty source window, and not just line art in target window?
                        raise Exception("target window does not fit the source window in alignment",
                                        line, source_windows, target_windows)
                    if (strict and j == 0 and
                        source_windows[-1].strip(u'—-. ')):
                        # empty target window, and not just line art in source window?
                        raise Exception("source window does not fit the target window in alignment",
                                        line, source_windows, target_windows)
                    if i > self.window_length:
                        raise Exception("source window too long", i, j,
                                        source_windows, target_windows)
                    if j > self.window_length:
                        raise Exception("target window too long", i, j,
                                        source_windows, target_windows)
                    # make new window
                    if (i > 0 and j > 0 and
                        (source_char == self.aligner.gap_element and u'—-. '.find(target_char) < 0 or
                         target_char == self.aligner.gap_element and u'—-. '.find(source_char) < 0)):
                        # move last character from both current windows to new ones
                        source_windows.append(source_windows[-1][-1:])
                        target_windows.append(target_windows[-1][-1:])
                        source_windows[-2] = source_windows[-2][:-1]
                        target_windows[-2] = target_windows[-2][:-1]
                        i = 1
                        j = 1
                        if with_confidence: # binary input with OCR confidence?
                            sourceconf_windows.append(sourceconf_windows[-1][-1:])
                            sourceconf_windows[-2] = sourceconf_windows[-2][:-1]
                    else:
                        i = 0
                        j = 0
                        source_windows.append('')
                        target_windows.append('')
                        if with_confidence: # binary input with OCR confidence?
                            sourceconf_windows.append([])
                if source_char != self.aligner.gap_element:
                    source_windows[-1] += source_char
                    i += 1
                    if with_confidence: # binary input with OCR confidence?
                        sourceconf_windows[-1] += [source_conf]
                if target_char != self.aligner.gap_element:
                    target_windows[-1] += target_char
                    j += 1
        except Exception as exc:
            if verbose:
                self.logger.error('%s' 'windowing error: %s',
                                  '\x1b[2K\x1b[G', str(exc))
            # rid of the offending window, but keep the previous ones:
            source_windows.pop()
            target_windows.pop()
            if with_confidence: # binary input with OCR confidence?
                sourceconf_windows.pop()
        return source_windows, target_windows, sourceconf_windows
    
    def vectorize_windows(self, encoder_input_sequences, decoder_input_sequences, encoder_conf_sequences=None):
        # Note: padding and confidence indexing need Dense/dot instead of Embedding/gather.
        # Used both for training (teacher forcing) and inference (ignore decoder input/output/weights).
        # Special cases:
        # - true zero (no index): padding for encoder input,
        #                         start "symbol" for decoder input
        # - tab character: padding for decoder input and decoder output, not allowed in encoder
        # - empty character (index zero): underspecified encoder input, not allowed in decoder
        encoder_input_data  = np.zeros((self.batch_size, self.window_length, self.voc_size),
                                       dtype=np.float32 if encoder_conf_sequences else np.uint32)
        decoder_input_data  = np.zeros((self.batch_size, self.window_length+1, self.voc_size), dtype=np.uint32)
        decoder_output_data = np.zeros((self.batch_size, self.window_length+1, self.voc_size), dtype=np.uint32)
        for i, (enc_seq, dec_seq) in enumerate(zip(encoder_input_sequences, decoder_input_sequences)):
            if i >= self.batch_size:
                raise Exception('input sequences %d (%s\t%s) exceed batch size', i, enc_seq, dec_seq)
            j = 0 # to declare scope outside loop
            for j, char in enumerate(enc_seq):
                if j >= self.window_length:
                    raise Exception('encoder input sequence %d (%s) exceeds window length', i, enc_seq)
                if char not in self.mapping[0]:
                    self.logger.error('unmapped character "%s" at encoder input sequence %d', char, i)
                    idx = 0 # underspecification
                elif char == '\t':
                    raise Exception('encoder input sequence %d (%s) contains tab character', i, enc_seq)
                else:
                    idx = self.mapping[0][char]
                encoder_input_data[i, j, idx] = 1
                if encoder_conf_sequences: # binary input with OCR confidence?
                    encoder_input_data[i, j, idx] = encoder_conf_sequences[i][j]
            # ...other j for encoder input: padding (keep zero)
            # j == 0 for decoder input: start symbol (keep zero)
            for j, char in enumerate(dec_seq):
                if j >= self.window_length:
                    raise Exception('decoder input sequence %d (%s) exceeds window length', i, dec_seq)
                if char not in self.mapping[0]:
                    self.logger.error('unmapped character "%s" at decoder input sequence %d', char, i)
                    idx = 0
                else:
                    idx = self.mapping[0][char]
                decoder_input_data[i, j+1, idx] = 1
                # teacher forcing:
                decoder_output_data[i, j, idx] = 1
            # j == len(dec_seq) for decoder output: padding
            idx = self.mapping[0]['\t']
            decoder_output_data[i, j+1, idx] = 1
            # ...other j for decoder input and output: padding:
            for j in range(self.window_length, j+1):
                decoder_input_data[i, j+1, idx] = 1
                decoder_output_data[i, j+1, idx] = 1
        
        # index of padded samples, so we can mask them
        # with the sample_weight parameter during fit() below
        #decoder_output_weights = np.ones(decoder_output_data.shape[:-1], dtype=np.float32)
        #decoder_output_weights[np.all(decoder_output_data == 0, axis=2)] = 0. # true zero (empty window)
        if self.lm_loss:
            # 2 outputs, 1 combined loss:
            decoder_output_data = [decoder_output_data, decoder_output_data]
            #decoder_output_weights = [decoder_output_weights, lm_output_weights]
        
        return encoder_input_data, decoder_input_data, decoder_output_data#, decoder_output_weights
    
    def save(self, filename):
        '''Save model weights and configuration parameters.

        Save configured model parameters into `filename`.
        (This preserves weights across CPU/GPU implementations or input shape configurations.)
        '''
        assert self.status > 1 # already trained
        self.logger.info('Saving model under "%s"', filename)
        self.encoder_decoder_model.save_weights(filename)
        with h5py.File(filename, 'a') as file:
            config = file.create_group('config')
            config.create_dataset('width', data=np.array(self.width))
            config.create_dataset('depth', data=np.array(self.depth))
            config.create_dataset('length', data=np.array(self.window_length))
            config.create_dataset('stateful', data=np.array(self.stateful))
            config.create_dataset('residual_connections', data=np.array(self.residual_connections))
            config.create_dataset('deep_bidirectional_encoder', data=np.array(self.deep_bidirectional_encoder))
            config.create_dataset('bridge_dense', data=np.array(self.bridge_dense))
            config.create_dataset('mapping',
                                  data=np.fromiter((ord(self.mapping[1][i])
                                                    if i in self.mapping[1] and self.mapping[1][i] else 0
                                                    for i in range(self.voc_size)), dtype=np.uint32))
    
    def load_config(self, filename):
        '''Load parameters to prepare configuration/compilation.

        Load model configuration from `filename`.
        '''
        with h5py.File(filename, 'r') as file:
            config = file['config']
            self.width = config['width'][()]
            self.depth = config['depth'][()]
            self.window_length = config['length'][()] \
                                 if 'length' in config else 7 # old default
            self.stateful = config['stateful'][()]
            self.residual_connections = config['residual_connections'][()] \
                                        if 'residual_connections' in config else False # old default
            self.deep_bidirectional_encoder = config['deep_bidirectional_encoder'][()] \
                                              if 'deep_bidirectional_encoder' in config else False # old default
            self.bridge_dense = config['bridge_dense'][()] \
                                if 'bridge_dense' in config else False # old default
            c_i = dict((chr(c), i) if c > 0 else ('', 0) for i, c in enumerate(config['mapping'][()]))
            i_c = dict((i, chr(c)) if c > 0 else (0, '') for i, c in enumerate(config['mapping'][()]))
            self.mapping = (c_i, i_c)
            self.voc_size = len(c_i)
    
    def load_weights(self, filename):
        '''Load weights into the configured/compiled model.

        Load weights from `filename` into the compiled and configured model.
        (This preserves weights across CPU/GPU implementations or input shape configurations.)
        '''
        assert self.status > 0 # already compiled
        self.logger.info('Loading model from "%s"', filename)
        self.encoder_decoder_model.load_weights(filename)
        self.status = 2
    
    def load_transfer_weights(self, filename):
        '''Load weights from another model into the configured/compiled model.

        Load weights from `filename` into the matching layers of the compiled and configured model.
        The other model need not have exactly the same configuration.
        (This preserves weights across CPU/GPU implementations or input shape configurations.)
        '''
        from keras.engine.saving import load_weights_from_hdf5_group_by_name
        assert self.status > 0 # already compiled
        assert self.depth > 1
        with h5py.File(filename, mode='r') as file:
            if 'layer_names' not in file.attrs and 'model_weights' in file:
                file = file['model_weights']
            was_shallow = False
            if 'config' in file:
                config = file['config']
                c_i = dict((chr(c), i) if c > 0 else ('', 0) for i, c in enumerate(config['mapping'][()]))
                i_c = dict((i, chr(c)) if c > 0 else (0, '') for i, c in enumerate(config['mapping'][()]))
                self.mapping = (c_i, i_c)
                self.voc_size = len(c_i)
                self.reconfigure_for_mapping()
                if config['depth'][()] == self.depth - 1:
                    was_shallow = True
            self.logger.info('Transferring model from "%s"', filename)
            load_weights_from_hdf5_group_by_name(file, self.encoder_decoder_model.layers,
                                                 skip_mismatch=True, reshape=False)
            if was_shallow:
                self.logger.info('fixing weights from shallower model')
                for i in range(1, self.depth): # fix previous layer weights
                    self.encoder_decoder_model.get_layer(name='encoder_lstm_%d'%i).trainable = False
                    self.encoder_decoder_model.get_layer(name='decoder_lstm_%d'%i).trainable = False
                self.recompile() # necessary for trainable to take effect
        self.status = 1

    def decode_batch_greedy(self, encoder_input_data):
        '''Predict from one batch array source window without alternatives.
        
        Use encoder input window array `encoder_input_data` (in a full batch).
        Start decoder with start-of-sequence, then keep decoding until
        end-of-sequence is found or output window length is full.
        Decode by using the best predicted output character as next input.
        Pass decoder initial/final state from character to character.
        '''
        
        states_value = self.encoder_model.predict_on_batch(encoder_input_data)
        decoder_input_data = np.zeros((self.batch_size, 1, self.voc_size), dtype=np.uint32)
        decoder_output_data = np.zeros((self.batch_size, self.window_length+1, self.voc_size), dtype=np.uint32)
        decoder_input_data[:, 0, self.mapping[0]['\t']] = 1
        for i in range(self.window_length+1):
            decoder_output_data[:, i] = decoder_input_data[:, -1]
            output = self.decoder_model.predict_on_batch([decoder_input_data] + states_value)
            output_scores = output[0]
            output_states = output[1:]
            # if sampling from the raw distribution, we could stop here
            indexes = np.nanargmax(output_scores, axis=2)
            decoder_input_data = np.eye(self.voc_size, dtype=np.uint32)[indexes]
            states_value = list(output_states)
        return decoder_output_data
    
    def decode_sequence_greedy(self, source_seq=None, source_state=None):
        '''Predict from one line vector source window without alternatives.
        
        Use encoder input window vector `source_seq` (in a batch of size 1).
        If `source_state` is given, bypass that step to protect the encoder state.
        Start decoder with start-of-sequence, then keep decoding until
        end-of-sequence is found or output window length is full.
        Decode by using the best predicted output character as next input.
        Pass decoder initial/final state from character to character.
        '''
        # reset must be done at line break (by caller)
        #self.encoder_model.reset_states()
        #self.decoder_model.reset_states()
       
        # Encode the source as state vectors.
        states_value = source_state if source_state is not None else \
            self.encoder_model.predict_on_batch(np.expand_dims(source_seq, axis=0))
        
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.voc_size), dtype=np.uint32)
        # The first character (start symbol) stays empty.
        #target_seq[0, 0, self.mapping[0]['\t']] = 1.
        
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        decoded_text = ''
        for i in range(1, self.window_length+1):
            output = self.decoder_model.predict_on_batch([target_seq] + states_value)
            output_scores = output[0]
            output_states = output[1:]
            
            # Sample a token:
            idx = np.nanargmax(output_scores[0, -1, :])
            char = self.mapping[1][idx]
            if char == '': # underspecification
                output_scores[0, -1, idx] = np.nan
                idx = np.nanargmax(output_scores[0, -1, :])
                char = self.mapping[1][idx]
            # Exit condition: either hit max length or find padding character.
            if char == '\t':
                break
            decoded_text += char
            if char == '\n':
                break
            
            # Update the target sequence (of length 1):
            target_seq = np.eye(self.voc_size, dtype=np.uint32)[[[[idx]]]]
            # Update states:
            states_value = list(output_states)
        
        return decoded_text
    
    def decode_sequence_beam(self, source_seq=None, source_state=None, eol=False):
        '''Predict from one line vector source window with alternatives.
        
        Use encoder input window vector `source_seq` (in a batch of size 1).
        If `source_state` is given, bypass that step to protect the encoder state.
        Start decoder with start-of-sequence, then keep decoding until
        end-of-sequence is found or output window length is full, repeatedly
        (but at most beam_width_out times or a maximum number of steps).
        Decode by using the best predicted output characters and several next-best
        alternatives (up to some degradation threshold) as next input.
        Follow-up on the n-best overall candidates (estimated by accumulated
        score, normalized by length), i.e. do breadth-first search.
        Pass decoder initial/final state from character to character,
        for each candidate respectively.
        '''
        from bisect import insort_left
        
        # reset must be done at line break (by caller)
        #self.encoder_model.reset_states()
        #self.decoder_model.reset_states()
        next_fringe = [Node(state=source_state
                            if source_state is not None else self.encoder_model.predict_on_batch(
                                    np.expand_dims(source_seq, axis=0)), # layer list of state arrays
                            value='',
                            cost=0.0)]
        hypotheses = []
        # generator will raise StopIteration if no hypotheses after loop
        max_batches = self.window_length*3 # how many batches (i.e. char hypotheses) will be processed per window?
        for l in range(max_batches):
            # try:
            #     next(n for n in next_fringe if all(np.array_equal(x,y) for x,y in zip(nonbeam_states[:l+1], [s.state for s in n.to_sequence()])))
            # except StopIteration:
            #     print('greedy result falls off beam search at pos', l, nonbeam_seq[:l+1])
            fringe = []
            while next_fringe:
                node = next_fringe.pop()
                if node.value == '\t': # padding symbol?
                    insort_left(hypotheses, node.parent)
                    #self.logger.debug('found new solution by padding "%s"', ''.join(node.to_sequence_of_values()))
                elif node.value == '\n': # end-of-sequence symbol?
                    insort_left(hypotheses, node)
                    #self.logger.debug('found new solution by newline "%s"', ''.join(node.to_sequence_of_values()))
                elif node.length > self.window_length+1: # window full?
                    insort_left(hypotheses, node)
                    #self.logger.debug('found new solution by length "%s"', ''.join(node.to_sequence_of_values()))
                else: # normal step
                    fringe.append(node)
                    #print('added new hypothesis "%s"' % bytes([i for i in n.to_sequence_of_values()]).decode("utf-8", "ignore"))
                if len(fringe) >= self.batch_size:
                    break # enough for one batch
            if not fringe:
                break # will give StopIteration unless we have some results already
            if len(hypotheses) > self.beam_width_out:
                break # it is unlikely that later iterations will find better top n results
            
            # use fringe leaves as minibatch, but with only 1 timestep
            if l == 0: # start symbol is true zero
                target_seq = np.zeros((len(fringe), 1, self.voc_size), dtype=np.uint32)
            else:
                target_seq = np.expand_dims(
                    np.eye(self.voc_size, dtype=np.uint32)[[self.mapping[0][node.value] for node in fringe]],
                    axis=1) # add time dimension
            states_val = [np.vstack([node.state[layer] for node in fringe])
                          for layer in range(len(fringe[0].state))] # stack layers across batch
            output = self.decoder_model.predict_on_batch([target_seq] + states_val)
            scores_output = output[0][:, -1] # only last timestep
            states_output = list(output[1:]) # from (layers) tuple
            for i, node in enumerate(fringe): # iterate over batch (1st dim)
                states = [layer[i:i+1] for layer in states_output] # unstack layers for current sample
                scores = scores_output[i]
                scores_order = np.argsort(scores) # still in reverse order (worst first)
                logscores = -np.log(scores[scores_order])
                highest = scores[scores_order[-1]]
                beampos = self.voc_size - np.searchsorted(scores[scores_order], highest / 3) # variable beam width
                #beampos = self.beam_width_in # fixed beam width
                pos = 0
                # follow up on best predictions, in true order (best first):
                for idx, logscore in zip(reversed(scores_order), reversed(logscores)):
                    value = self.mapping[1][idx]
                    if (np.isnan(logscore) or
                        (value == '\n' and not eol) or
                        value == ''):
                        continue # ignore this alternative
                    pos += 1
                    if pos > beampos:
                        break # ignore further alternatives
                    new_node = Node(parent=node, state=states,
                                    value=value, cost=logscore)
                    insort_left(next_fringe, new_node)
            if len(next_fringe) > max_batches * self.batch_size: # more than can ever be processed within limits?
                next_fringe = next_fringe[:max_batches*self.batch_size] # to save memory, keep only best
        for hypothesis in hypotheses[:self.beam_width_out]:
            yield ''.join(hypothesis.to_sequence_of_values())

class Node(object):
    def __init__(self, state, value, cost, parent=None, extras=None):
        super(Node, self).__init__()
        self.value = value # character
        self.parent = parent # parent Node, None for root
        self.state = state # recurrent layer hidden state
        self.cum_cost = parent.cum_cost + cost if parent else cost # e.g. -log(p) of sequence up to current node (including)
        self.length = 1 if parent is None else parent.length + 1
        self.extras = extras # node identifier
        self._sequence = None
    
    def to_sequence(self):
        # Return sequence of nodes from root to current node.
        if not self._sequence:
            self._sequence = []
            current_node = self
            while current_node:
                self._sequence.insert(0, current_node)
                current_node = current_node.parent
        return self._sequence
    
    def to_sequence_of_values(self):
        return [s.value for s in self.to_sequence()[1:]]
    
    # for sort order, use cumulative costs relative to length
    # (in order to get a fair comparison across different lengths,
    #  and hence, depth-first search), and use inverse order
    # (so the faster pop() can be used)
    def __lt__(self, other):
        return self.cum_cost / self.length < other.cum_cost / other.length
    def __le__(self, other):
        return self.cum_cost / self.length <= other.cum_cost / other.length
    def __eq__(self, other):
        return self.cum_cost / self.length == other.cum_cost / other.length
    def __ne__(self, other):
        return self.cum_cost / self.length != other.cum_cost / other.length
    def __gt__(self, other):
        return self.cum_cost / self.length > other.cum_cost / other.length
    def __ge__(self, other):
        return self.cum_cost / self.length >= other.cum_cost / other.length
