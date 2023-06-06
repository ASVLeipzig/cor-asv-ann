# -*- coding: utf-8
import unicodedata
import math
import logging
import pickle
import numpy as np
import h5py

from .alignment import Alignment, Edits

GAP = '\a' # reserved character that does not get mapped (for gap repairs)

class Sequence2Sequence(object):
    '''Sequence to sequence (character-level) error correction with Keras.

    Adapted from examples/lstm_seq2seq.py (tutorial by Francois Chollet
    "A ten-minute introduction...") with changes as follows:

    - use early stopping to prevent overfitting
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
      use newline character for decoder padding (learned/not masked in training,
      treated like end-of-sequence in inference)
    - add runtime preprocessing function for convenient single-line testing
    - change first layer to bidirectional, stack unidirectional LSTM layers
      on top (with HL depth and HL width configurable)
    - add beam search decoding (A*)
    - detect CPU vs GPU mode automatically
    - save/load weights separate from configuration (by recompiling model)
      in order to share weights between CPU and GPU model,
      and between fixed and variable batchsize/length
    - evaluate word and character error rates on separate dataset
    - use line-global additive-linear soft attention to connect encoder
      (top-HL) outputs to decoder (top-HL) inputs (instead of mere
      final-initial state transfer)
    - add topology variant: deep bi-directional encoder
    - add topology variant: residual connections
    - add topology variant: dense bridging final-initial state transfer
    - add training variant: scheduled sampling
    - add training variant: parallel LM loss
    - allow incremental training (e.g. pretraining on clean text)
    - allow weight transfer from shallower model (fixing shallow layer
      weights) or from language model (as unconditional decoder),
      update character mapping and re-use character embeddings
    - allow resetting encoder after load/init transfer
    
    Features still (very much) wanting of implementation:
    
    - stateful decoder mode (in non-transfer part of state function)
    - attention decoding with (linear-time) hard monotonic alignment
      instead of softmax alignment (with quadratic-time complexity)
    - context conditioning (with meta-data inputs like OCR engine)
    - methods to avoid exposure bias and label/myopic bias:
      generalized adversarial training (Huszár 2015),
      beam search optimization (Wiseman & Rush 2016),
      professor forcing (Lamb & Goyal et al 2016), or
      prospective performance network (Wang et al 2018)
    - systematic hyperparameter treatment (deviations from Sutskever
      should be founded on empirical analysis):
      HL width and depth, optimiser choice (RMSprop/SGD) and parameters,
      gradient clipping, decay and rate control, initialisers
    
    # Summary of the algorithm
    
    - In the learning phase, we have source sequences from OCR,
      and correspding target sequences from GT. We train:
      - a stacked LSTM encoder to turn the source sequences
        to output sequences and final hidden layer states.
      - a stacked LSTM decoder to turns the target sequences
        into the same sequence but offset by one timestep in the future,
        (a setup called "teacher forcing" in this context),
        based on the initial state vectors and the output sequences
        from the encoder.
      Effectively, the encoder-decoder learns to generate a sequence
      `targets[t+1...]` given `targets[...t]`, conditioned
      on the source sequence.
    - In inference mode, to decode unknown target sequences, we:
        - encode the source sequence into encoded sequence and state,
        - start with a target sequence of size 1
          (just the start-of-sequence character)
        - feed-back the state vectors and 1-character target sequence
          to the decoder to produce predictions for the next character
        - sample the next character using these predictions
          (using argmax for greedy and argsort for beam search)
        - append the sampled character to the target sequence
        - repeat until we generate the end-of-sequence character,
          or we hit a character length limit.
    
    # References
    
    - Sequence to Sequence Learning with Neural Networks
        https://arxiv.org/abs/1409.3215
    - Learning Phrase Representations using
        RNN Encoder-Decoder for Statistical Machine Translation
        https://arxiv.org/abs/1406.1078
    '''
    
    def __init__(self, logger=None, progbars=True):
        ### model parameters
        # How many samples are trained/decoded together (in parallel)?
        self.batch_size = 64
        # stateful decoder (implicit state transfer between batches)?
        self.stateful = False
        # number of nodes in the hidden layer (dimensionality of the encoding space)?
        self.width = 512
        # number of encoder and decoder layers stacked above each other?
        self.depth = 2
        # indexation of (known/allowed) input and output characters (i.e. vocabulary)
        #   note: character mapping includes nul for unknown/underspecification,
        #         and newline for end-of-sequence;
        #         mapping/voc_size is set by loading or training
        self.mapping = ({'': 0}, {0: ''})
        self.voc_size = 1 # size of mapping (0 reserved for unknown)
        # add input to output in each encoder and decoder hidden layer?
        self.residual_connections = False
        # encoder hidden layers are all bidirectional LSTMs,
        # cross-summarizing forward and backward outputs
        # (like -encoder_type bdrnn in Open-NMT)?
        self.deep_bidirectional_encoder = False
        # use a fully connected non-linear layer to transfer
        # encoder final states to decoder initial states instead of copy?
        self.bridge_dense = False
        
        ### training parameters
        # maximum number of epochs to train
        # (unless stopping early via validation loss)?
        self.epochs = 100
        # train with additional output (unweighted sum loss) from LM,
        # defined with tied decoder weights and same input, but
        # not conditioned on encoder output
        # (applies to encoder_decoder_model only, i.e. does not affect
        #  encoder_model and decoder_model during inference):
        self.lm_loss = False
        # predict likewise, and use during beam search such that
        # decoder scores control entry of local alternatives and
        # LM scores rate global alternatives of the beam
        # (applies to decoder_model only, but should be used on models
        #  with lm_loss during training):
        self.lm_predict = False
        # randomly train with decoder output from self-loop (softmax feedback)
        # instead of teacher forcing (with ratio given curve across epochs),
        # defined with tied weights and same encoder output
        # (applies to encoder_decoder_model only, i.e. does not affect
        #  encoder_model and decoder_model during inference)?
        self.scheduled_sampling = None # 'linear'/'sigmoid'/'exponential'/None
        # rate of dropped output connections in encoder and decoder HL?
        self.dropout = 0.2
        
        ### beam decoder inference parameters
        # probability of the input character candidate in each hypothesis
        # (unless already misaligned); helps balance precision/recall trade-off
        self.rejection_threshold = 0.3
        # up to how many new candidates can enter the beam per context/node?
        self.beam_width_in = 15
        # how much worse relative to the probability of the best candidate
        # may new candidates be to enter the beam?
        self.beam_threshold_in = 0.2
        # up to how many results can be drawn from result generator?
        self.beam_width_out = 16

        ### runtime variables
        self.logger = logger or logging.getLogger(__name__)
        self.graph = None # for tf access from multiple threads
        self.encoder_decoder_model = None # combined model for training
        self.encoder_model = None # separate model for inference
        self.decoder_model = None # separate model for inference (but see _resync_decoder)
        self.aligner = Alignment(0, logger=self.logger) # aligner (for training) with internal state
        self.progbars = progbars
        self.status = 0 # empty / configured / trained?
    
    def __repr__(self):
        return (__name__ +
                " (width: %d)" % self.width +
                " (depth: %d)" % self.depth +
                " (chars: %d)" % self.voc_size +
                " (attention)" +
                (" (stateful)" if self.stateful else " (stateless)") +
                " status: %s" % ("empty" if self.status < 1 else "configured" if self.status < 2 else "trained"))
    
    def configure(self, batch_size=None):
        '''Define encoder and decoder models for the configured parameters.
        
        Use given `batch_size` for encoder input if stateful:
        configure once for training phase (with parallel lines),
        then reconfigure for prediction (with only 1 line each).
        '''
        from keras.initializers import RandomNormal
        from keras.layers import Input, Dense, TimeDistributed, Dropout, Lambda
        from keras.layers import RNN, LSTMCell, LSTM, CuDNNLSTM, Bidirectional
        from keras.layers import concatenate, average, add
        from keras.models import Model
        #from keras.utils import plot_model
        from keras import backend as K
        import tensorflow as tf
        from .attention import DenseAnnotationAttention
        
        if batch_size:
            self.batch_size = batch_size

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        K.set_session(tf.compat.v1.Session(config=config))
        # self.sess = tf.compat.v1.Session()
        # K.set_session(self.sess)
        
        # automatically switch to CuDNNLSTM if CUDA GPU is available:
        has_cuda = K.backend() == 'tensorflow' and K.tensorflow_backend._get_available_gpus()
        self.logger.info('using %s LSTM implementation to compile %s model '
                         'of depth %d width %d size %d with attention',
                         'GPU' if has_cuda else 'CPU',
                         'stateful' if self.stateful else 'stateless',
                         self.depth, self.width, self.voc_size)
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
        
        # encoder part:
        encoder_input = Input(shape=(None, self.voc_size),
                              name='encoder_input')
        char_embedding = Dense(self.width, use_bias=False,
                               kernel_initializer=RandomNormal(stddev=0.001),
                               kernel_regularizer=self._regularise_chars,
                               name='char_embedding')
        char_input_proj = TimeDistributed(char_embedding, name='char_input_projection')
        encoder_output = char_input_proj(encoder_input)
        
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
        
        # Set up encoder HL to return output activation (to be attended to by decoder),
        # return final states as well (as initial states for the decoder).
        # Only the base hidden layer is bidirectional (unless deep_bidirectional_encoder).
        encoder_state_outputs = []
        for n in range(self.depth):
            args = {'name': 'encoder_lstm_%d' % (n+1),
                    'return_state': True,
                    'return_sequences': True}
            if not has_cuda:
                # instead of default 'hard_sigmoid' which deviates from CuDNNLSTM:
                args['recurrent_activation'] = 'sigmoid'
            layer = lstm(self.width, **args)
            if n == 0 or self.deep_bidirectional_encoder:
                encoder_output, fw_state_h, fw_state_c, bw_state_h, bw_state_c = (
                    Bidirectional(layer, name=layer.name)(
                        encoder_output if n == 0 else cross_sum(encoder_output)))
                # prepare for base layer decoder initial_state:
                # (the final states of the backward-LSTM, closest to the start of the line,
                #  in the encoder are used to initialise the state of the decoder)
                state_h = bw_state_h # ignore final fw state
                state_c = bw_state_c # ignore final fw state
            else:
                encoder_output2, state_h, state_c = layer(encoder_output)
                if self.residual_connections:
                    # add residual connections:
                    if n == 1:
                        #encoder_output = add([encoder_output2, average([encoder_output[:,:,::2], encoder_output[:,:,1::2]])]) # does not work (no _inbound_nodes)
                        encoder_output = encoder_output2
                    else:
                        encoder_output = add([encoder_output2, encoder_output])
                else:
                    encoder_output = encoder_output2
            constant_shape = (1, self.width * 2
                              if n == 0 or self.deep_bidirectional_encoder
                              else self.width)
            # variational dropout (time-constant) – LSTM (but not CuDNNLSTM)
            # has the (non-recurrent) dropout keyword option for this:
            encoder_output = Dropout(self.dropout, noise_shape=constant_shape)(encoder_output)
            if self.bridge_dense:
                state_h = Dense(self.width, activation='tanh', name='bridge_h_%d' % (n+1))(state_h)
                state_c = Dense(self.width, activation='tanh', name='bridge_c_%d' % (n+1))(state_c)
            encoder_state_outputs.extend([state_h, state_c])

        # just for convenience:
        # include zero as initial attention state in encoder state output
        # (besides final encoder state as initial cell state):
        attention_init = Lambda(lambda x: K.zeros_like(x)[:, :, 0],
                                name='attention_state_init')
        encoder_state_outputs.append(attention_init(encoder_output))
        # decoder-independent half of the encoder annotation
        # can be computed for the complete encoder sequence
        # at once (independent of the RNN state):
        attention_dense = TimeDistributed(Dense(self.width, use_bias=False),
                                          name='attention_dense')
        
        # decoder part:
        decoder_input = Input(shape=(None, self.voc_size),
                              name='decoder_input')
        decoder_input0 = char_input_proj(decoder_input)
        decoder_output = decoder_input0
        if self.lm_loss:
            lm_output = decoder_input0
        
        # Set up decoder HL to return full output sequences (so we can train in parallel),
        # to use encoder_state_outputs as initial state and return final states as well.
        # We don't use those states in the training model, but will use them for inference
        # (see further below).
        decoder_lstms = []
        for n in range(self.depth):
            args = {'name': 'decoder_lstm_%d' % (n+1),
                    'return_state': True,
                    'return_sequences': True}
            if n < self.depth - 1:
                if not has_cuda:
                    # instead of default 'hard_sigmoid' which deviates from CuDNNLSTM:
                    args['recurrent_activation'] = 'sigmoid'
                layer = lstm(self.width, **args)
                decoder_output2, _, _ = layer(decoder_output,
                                              initial_state=encoder_state_outputs[2*n:2*n+2])
                if self.lm_loss:
                    lm_output, _, _ = layer(lm_output)
            else:
                cell = DenseAnnotationAttention(
                    LSTMCell(self.width,
                             dropout=self.dropout,
                             recurrent_activation='sigmoid'),
                    window_width=5, # use local attention with 10 characters context
                    input_mode="concatenate",  # concat(input, context) when entering cell
                    output_mode="cell_output") # drop context when leaving cell
                layer = RNN(cell, **args)
                decoder_output2, _, _, _ = layer(decoder_output,
                                                 initial_state=encoder_state_outputs[2*n:2*n+3],
                                                 constants=[encoder_output,
                                                            attention_dense(encoder_output)])
                if self.lm_loss:
                    lm_output, _, _, _ = layer(lm_output)
            decoder_lstms.append(layer)
            # add residual connections:
            if n > 0 and self.residual_connections:
                decoder_output = add([decoder_output2, decoder_output])
            else:
                decoder_output = decoder_output2
            if n < self.depth - 1: # only hidden-to-hidden layer:
                constant_shape = (1, self.width)
                # variational dropout (time-constant) – LSTM (but not CuDNNLSTM)
                # has the (non-recurrent) dropout keyword option for this:
                decoder_output = Dropout(self.dropout, noise_shape=constant_shape)(decoder_output)
        
        def char_embedding_transposed(x):
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
        char_output_proj = TimeDistributed(Lambda(char_embedding_transposed, name='transpose+softmax'),
                                          name='char_output_projection')
        decoder_output = char_output_proj(decoder_output)
        if self.lm_loss:
            lm_output = char_output_proj(lm_output)
            decoder_output = [decoder_output, lm_output] # 2 outputs, 1 combined loss
        
        # Bundle the model that will turn
        # `encoder_input_data` and `decoder_input_data` into `decoder_output_data`
        self.encoder_decoder_model = Model([encoder_input, decoder_input], decoder_output,
                                           name='encoder_decoder_model')
        
        ## Define inference phase model:
        # 1) encode source to retrieve output sequence
        #    (attended) and initial decoder states
        #    (bw h/c, h/c, attention state)
        # 2) run one step of decoder with this initial state
        #    and a "start of sequence" as target token.
        # 3) repeat from 2, feeding back the target token
        #    from output to input, and passing states
        
        # Re-use the training phase encoder unchanged
        # (with sequence and final states as output):
        self.encoder_model = Model(
            encoder_input,
            [encoder_output] + encoder_state_outputs,
            name='encoder_model')
        
        # Set up decoder differently:
        # - with additional input for encoder output
        #   (attended sequence)
        # - with additional input for initial states
        #   (not just encoder_state_outputs at first step)
        # - keeping and concatenating final states
        #   (instead of discarding)
        # so we can pass states explicitly:
        decoder_state_inputs = []
        decoder_state_outputs = []
        decoder_output = decoder_input0
        if self.lm_predict:
            lm_output = decoder_input0
        for n in range(self.depth):
            state_h_in = Input(shape=(self.width,),
                               name='initial_h_%d_input' % (n+1))
            state_c_in = Input(shape=(self.width,),
                               name='initial_c_%d_input' % (n+1))
            decoder_state_inputs.extend([state_h_in, state_c_in])
            layer = decoder_lstms[n] # tied weights
            if n < self.depth - 1:
                decoder_output, state_h_out, state_c_out = layer(
                    decoder_output,
                    initial_state=decoder_state_inputs[2*n:2*n+2])
                decoder_state_outputs.extend([state_h_out,
                                              state_c_out])
                if self.lm_predict:
                    lm_output, _, _ = layer(
                        lm_output,
                        initial_state=decoder_state_inputs[2*n:2*n+2])
            else:
                attention_input = Input(shape=(None, self.width),
                                        name='attention_input')
                attention_state_in = Input(shape=(None,),
                                           name='attention_state_input')
                decoder_state_inputs.append(attention_state_in)
                # for some obscure reason, layer sharing is impossible
                # with DenseAnnotationAttention; so we must redefine
                # and then resync weights after training/loading
                # (see _resync_decoder):
                cell = DenseAnnotationAttention(
                    LSTMCell(self.width,
                             dropout=self.dropout,
                             recurrent_activation='sigmoid'),
                    window_width=5, # use local attention with 10 characters context
                    input_mode="concatenate",  # concat(input, context) when entering cell
                    output_mode="cell_output") # drop context when leaving cell
                layer = RNN(cell, **args)
                decoder_output, state_h_out, state_c_out, attention_state_out = layer(
                    decoder_output,
                    initial_state=decoder_state_inputs[2*n:2*n+3],
                    constants=[attention_input,
                               attention_dense(attention_input)])
                decoder_state_outputs.extend([state_h_out,
                                              state_c_out,
                                              attention_state_out])
                if self.lm_predict:
                    attention_zero = Lambda(lambda x: K.zeros_like(x))(attention_input)
                    lm_output, _, _, _ = layer(
                        lm_output,
                        initial_state=decoder_state_inputs[2*n:2*n+3],
                        constants=[attention_zero, attention_zero])
        decoder_output = char_output_proj(decoder_output)
        if self.lm_predict:
            lm_output = char_output_proj(lm_output)
            decoder_output = [decoder_output, lm_output] # 2 outputs (1 for local, 1 for global scores)
        else:
            decoder_output = [decoder_output]
        # must be resynced each time encoder_decoder_model changes:
        self.decoder_model = Model(
            [decoder_input, attention_input] + decoder_state_inputs,
            decoder_output + decoder_state_outputs,
            name='decoder_model')
        
        ## Compile model
        self._recompile()
        # for tf access from multiple threads
        # self.encoder_model._make_predict_function()
        # self.decoder_model._make_predict_function()
        # self.sess.run(tf.global_variables_initializer())
        self.graph = tf.compat.v1.get_default_graph()
        self.status = 1
    
    def _recompile(self):
        from keras.optimizers import Adam
        
        self.encoder_decoder_model.compile(
            loss='categorical_crossentropy', # loss_weights=[1.,1.] if self.lm_loss
            optimizer=Adam(clipnorm=5), #'adam',
            sample_weight_mode='temporal') # sample_weight slows down training slightly (20%)
    
    def _reconfigure_for_mapping(self):
        '''Reconfigure character embedding layer after change of mapping (possibly transferring previous weights).'''
        
        assert self.status >= 1
        embedding = self.encoder_decoder_model.get_layer(name='char_input_projection').layer # cannot get char_embedding directly
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
                    if layer.name == 'char_input_projection':
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
    def _resync_decoder(self):
        self.decoder_model.get_layer('decoder_lstm_%d' % self.depth).set_weights(
            self.encoder_decoder_model.get_layer('decoder_lstm_%d' % self.depth).get_weights())
    
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
        
        return K.in_train_phase(lowrank + underspecification, 0.)
    
    def map_files(self, filenames):
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
                        if not source_conf: # empty
                            line = target_text
                        elif type(source_conf[0]) is tuple: # prob line
                            line = ''.join([char for char, prob in source_conf]) + target_text
                        else: # confmat
                            line = ''.join([chars for chunk in source_conf
                                            for chars, prob in chunk]) + target_text
                    line = unicodedata.normalize('NFC', line)
                    chars.update(set(line))
                    if GAP in chars:
                        self.logger.warning('ignoring gap character "%s" in input file "%s"', GAP, filename)
                        chars.remove(GAP)
                    num_lines += 1
        chars = sorted(list(chars))
        if len(chars) > self.voc_size:
            # incremental training
            c_i = dict((c, i) for i, c in enumerate(chars))
            i_c = dict((i, c) for i, c in enumerate(chars))
            self.mapping = (c_i, i_c)
            self.voc_size = len(c_i)
            self._reconfigure_for_mapping()
        return num_lines
    
    def train(self, filenames, val_filenames=None):
        '''train model on given text files.
        
        Pass the character sequences of lines in `filenames`, paired into
        source and target (and possibly, source confidence values),
        to the loop training model weights with stochastic gradient descent.
        
        The generator will open each file, looping over the complete set (epoch)
        as long as validation error does not increase in between (early stopping).
        
        Validate on a random fraction of lines automatically separated before,
        unless `val_filenames` is given, in which case only those files are used
        for validation.
        '''
        from keras.callbacks import EarlyStopping, TerminateOnNaN
        from .callbacks import StopSignalCallback, ResetStatesCallback
        from .keras_train import fit_generator_autosized, evaluate_generator_autosized

        num_lines = self.map_files(filenames)
        self.logger.info('Training on "%d" files with %d lines', len(filenames), num_lines)
        if val_filenames:
            num_lines = self.map_files(val_filenames)
            self.logger.info('Validating on "%d" files with %d lines', len(val_filenames), num_lines)
            split_rand = None
        else:
            self.logger.info('Validating on random 20% lines from those files')
            split_rand = np.random.uniform(0, 1, (num_lines,)) # reserve split fraction at random line numbers
        
        # Run training
        earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1,
                                      mode='min', restore_best_weights=True)
        callbacks = [earlystopping, TerminateOnNaN(),
                     StopSignalCallback(logger=self.logger)]
        history = fit_generator_autosized(
            self.encoder_decoder_model,
            self.gen_data(filenames, split_rand, train=True),
            epochs=self.epochs,
            workers=1,
            # (more than 1 would effectively increase epoch size)
            use_multiprocessing=not self.scheduled_sampling,
            # (cannot access session/graph for scheduled sampling in other process,
            #  cannot access model for reset callback in other process)
            validation_data=self.gen_data(val_filenames or filenames, split_rand, train=False),
            verbose=1 if self.progbars else 0,
            callbacks=callbacks)
        
        if 'val_loss' in history.history:
            self.logger.info('training finished with val_loss %f',
                             min(history.history['val_loss']))
            if (np.isnan(history.history['val_loss'][-1]) or
                earlystopping.stopped_epoch == 0):
                # recover weights (which TerminateOnNaN prevented EarlyStopping from doing)
                self.encoder_decoder_model.set_weights(earlystopping.best_weights)
            self._resync_decoder()
            self.status = 2
        else:
            self.logger.critical('training failed')
            self.status = 1
    
    def evaluate(self, filenames, fast=False, normalization='historic_latin', charmap={}, gt_level=1,
                 confusion=10, histogram=True):
        '''evaluate model on text files
        
        Pass the character sequence of lines in ``filenames``, paired into
        source and target (and possibly, source confidence values),
        to a loop predicting outputs with decoder feedback and greedy+beam search.
        
        The generator will open each file, looping over the complete set once,
        printing source/target and predicted lines,
        and the overall calculated character and word error rates of source (OCR)
        and prediction (greedy/beamed) against target (GT).
        
        If ``fast``, then skip beam search for single lines, and process all batches
        in parallel greedily.
        
        For ``normalization`` and ``gt_level``, see ``Alignment.get_adjusted_distance()``.
        
        If ``charmap`` is non-empty, use it (as in str.maketrans) before processing.
        
        If ``confusion`` is greater than zero, then aggregate (non-identity) edits
        on the character level, and show this many most-frequent confusions in the end.
        '''
        # FIXME: stop using both greedy and beamed in 1 function
        assert self.status == 2
        c_origin_counts = Edits(self.logger, histogram=histogram)
        w_origin_counts = Edits(self.logger)
        c_greedy_counts = Edits(self.logger, histogram=histogram)
        w_greedy_counts = Edits(self.logger)
        c_beamed_counts = Edits(self.logger, histogram=histogram)
        w_beamed_counts = Edits(self.logger)
        c_origin_aligner = Alignment(0, logger=self.logger, confusion=confusion > 0)
        w_origin_aligner = Alignment(0, logger=self.logger)
        c_greedy_aligner = Alignment(0, logger=self.logger, confusion=confusion > 0)
        w_greedy_aligner = Alignment(0, logger=self.logger)
        c_beamed_aligner = Alignment(0, logger=self.logger, confusion=confusion > 0)
        w_beamed_aligner = Alignment(0, logger=self.logger)
        for batch_no, batch in enumerate(self.gen_lines(filenames, False, charmap=charmap)):
            lines_source, lines_sourceconf, lines_target, lines_filename = batch
            #bar.update(1)

            lines_greedy, probs_greedy, scores_greedy, _ = (
                self.correct_lines(lines_source, lines_sourceconf,
                                   fast=fast, greedy=True))
            if fast:
                lines_beamed, probs_beamed, scores_beamed = (
                    lines_greedy, probs_greedy, scores_greedy)
            else:
                lines_beamed, probs_beamed, scores_beamed, _ = (
                    self.correct_lines(lines_source, lines_sourceconf,
                                       fast=False, greedy=False))
            for j in range(len(lines_source)):
                if not lines_source[j] or not lines_target[j]:
                    continue # from partially filled batch
                
                self.logger.info('Source input              : %s',
                                 lines_source[j].rstrip(u'\n'))
                self.logger.info('Target output             : %s',
                                 lines_target[j].rstrip(u'\n'))
                self.logger.info('Target prediction (greedy): %s [%.2f]',
                                 lines_greedy[j].rstrip(u'\n'), scores_greedy[j])
                self.logger.info('Target prediction (beamed): %s [%.2f]',
                                 lines_beamed[j].rstrip(u'\n'), scores_beamed[j])
                
                #metric = get_levenshtein_distance

                c_origin_dist = c_origin_aligner.get_adjusted_distance(lines_source[j], lines_target[j],
                                                                       normalization=normalization,
                                                                       gtlevel=gt_level)
                c_greedy_dist = c_greedy_aligner.get_adjusted_distance(greedy_lines[j], target_lines[j],
                                                                       normalization=normalization,
                                                                       gtlevel=gt_level)
                c_beamed_dist = c_beamed_aligner.get_adjusted_distance(beamed_lines[j], target_lines[j],
                                                                       normalization=normalization,
                                                                       gtlevel=gt_level)
                c_origin_counts.add(c_origin_dist, lines_source[j], lines_target[j])
                c_greedy_counts.add(c_greedy_dist, lines_greedy[j], lines_target[j])
                c_beamed_counts.add(c_beamed_dist, lines_beamed[j], lines_target[j])
                
                tokens_greedy = lines_greedy[j].split(" ")
                tokens_beamed = lines_beamed[j].split(" ")
                tokens_source = lines_source[j].split(" ")
                tokens_target = lines_target[j].split(" ")
                
                w_origin_dist = w_origin_aligner.get_adjusted_distance(tokens_source, tokens_target,
                                                                       normalization=normalization,
                                                                       gtlevel=gt_level)
                w_greedy_dist = w_greedy_aligner.get_adjusted_distance(tokens_greedy, tokens_target,
                                                                       normalization=normalization,
                                                                       gtlevel=gt_level)
                w_beamed_dist = w_beamed_aligner.get_adjusted_distance(tokens_beamed, tokens_target,
                                                                       normalization=normalization,
                                                                       gtlevel=gt_level)
                w_origin_counts.add(w_origin_dist, tokens_source, tokens_target)
                w_greedy_counts.add(w_greedy_dist, tokens_greedy, tokens_target)
                w_beamed_counts.add(w_beamed_dist, tokens_beamed, tokens_target)
                
            c_greedy_counts.score += sum(scores_greedy)
            c_beamed_counts.score += sum(scores_beamed)

        self.logger.info('finished %d lines', c_origin_counts.length)
        if confusion > 0:
            self.logger.info('OCR    confusion: %s', c_origin_aligner.get_confusion(confusion))
            self.logger.info('greedy confusion: %s', c_greedy_aligner.get_confusion(confusion))
            self.logger.info('beamed confusion: %s', c_beamed_aligner.get_confusion(confusion))
        if histogram:
            self.logger.info('OCR    histogram: %s', repr(c_origin_counts.hist()))
            self.logger.info('greedy histogram: %s', repr(c_greedy_counts.hist()))
            self.logger.info('beamed histogram: %s', repr(c_beamed_counts.hist()))
        self.logger.info('ppl greedy: %.3f', math.exp(c_greedy_counts.score/c_greedy_counts.length))
        self.logger.info('ppl beamed: %.3f', math.exp(c_beamed_counts.score/c_beamed_counts.length))
        self.logger.info("CER OCR:    %.3f±%.3f", c_origin_counts.mean, math.sqrt(c_origin_counts.varia))
        self.logger.info("CER greedy: %.3f±%.3f", c_greedy_counts.mean, math.sqrt(c_greedy_counts.varia))
        self.logger.info("CER beamed: %.3f±%.3f", c_beamed_counts.mean, math.sqrt(c_beamed_counts.varia))
        self.logger.info("WER OCR:    %.3f±%.3f", w_origin_counts.mean, math.sqrt(w_origin_counts.varia))
        self.logger.info("WER greedy: %.3f±%.3f", w_greedy_counts.mean, math.sqrt(w_greedy_counts.varia))
        self.logger.info("WER beamed: %.3f±%.3f", w_beamed_counts.mean, math.sqrt(w_beamed_counts.varia))
        
    def predict(self, filenames, fast=False, greedy=False, charmap={}):
        '''apply model on text files
        
        Pass the character sequence of lines in ``filenames``, paired into
        source and target (and possibly, source confidence values),
        to a loop predicting outputs with decoder feedback and greedy/beam search.
        
        The generator will open each file, looping over the complete set once,
        yielding predicted lines (along with their filename).
        
        If ``fast``, then skip beam search for single lines, and process all batches
        in parallel greedily.
        
        If ``charmap`` is non-empty, use it (as in str.maketrans) before processing.
        '''
        assert self.status == 2
        for batch_no, batch in enumerate(self.gen_lines(filenames,
                                                        repeat=False,
                                                        unsupervised=True,
                                                        charmap=charmap)):
            lines_source, lines_sourceconf, _, lines_filename = batch
            lines_result, probs_result, scores_result, _ = (
                self.correct_lines(lines_source, lines_sourceconf,
                                   fast=fast, greedy=greedy))
            yield (lines_filename, lines_result, scores_result)

    def correct_lines(self, lines, conf=None, fast=True, greedy=True):
        '''apply correction model on text strings
        
        Pass the character sequences `lines` (optionally complemented by
        respective confidence values), to a loop predicting outputs with
        decoder feedback and greedy or beam search. Each line must end
        with a newline character.
        
        If `fast`, process all lines in parallel and all characters at once
        greedily.
        Otherwise, if `greedy`, process each line greedily (i.e. without
        beam search).
        
        Return a 4-tuple of the corrected lines, probability lists,
        perplexity scores, and input-output alignments.
        '''
        assert not fast or greedy, "cannot decode in fast mode with beam search enabled"
        if not lines:
            return [], [], [], []
        
        # vectorize:
        encoder_input_data, _, _, _ = self.vectorize_lines(lines, lines, conf)

        if fast:
            # encode and decode in batch (all lines at once):
            _, output_lines, output_probs, output_scores, alignments = self.decode_batch_greedy(encoder_input_data)
        else:
            # encode lines in batch (all lines at once):
            encoder_outputs = self.encoder_model.predict_on_batch(encoder_input_data)
            # decode lines and characters individually:
            output_lines, output_probs, output_scores, alignments = [], [], [], []
            for j, input_line in enumerate(lines):
                if not input_line:
                    line, probs, score, alignment = '', [], 0, []
                elif greedy:
                    line, probs, score, alignment = self.decode_sequence_greedy(
                        encoder_outputs=[encoder_output[j:j+1] for encoder_output in encoder_outputs])
                else:
                    # query only 1-best
                    try:
                        line, probs, score, alignment = next(self.decode_sequence_beam(
                            source_seq=encoder_input_data[j], # needed for rejection fallback
                            encoder_outputs=[encoder_output[j:j+1] for encoder_output in encoder_outputs]))
                    except StopIteration:
                        self.logger.error('cannot beam-decode input line %d: "%s"', j, input_line)
                        line = input_line
                        probs = [1.0] * len(line)
                        score = 0
                        alignment = np.eye(len(line)).tolist()
                line = line.replace(GAP, '') # remove if rejected (i.e. not corrected despite underspecification)
                output_lines.append(line)
                output_probs.append(probs)
                output_scores.append(score)
                alignments.append(alignment)
        return output_lines, output_probs, output_scores, alignments
    
    # for fit_generator()/predict_generator()/evaluate_generator()/standalone
    # -- looping, but not shuffling
    def gen_data(self, filenames, split=None, train=False, unsupervised=False, charmap={}, reset_cb=None):
        '''generate batches of vector data from text file
        
        Open `filenames` in text mode, loop over them producing `batch_size`
        lines at a time. Pad lines into the longest line of the batch.
        If stateful, call `reset_cb` at the start of each batch (if given)
        or reset model directly (otherwise).
        Skip lines at `split` positions (if given), depending on `train`
        (upper vs lower partition).
        Yield vector data batches (for fit_generator/evaluate_generator).
        '''
        
        epoch = 0
        if train and self.scheduled_sampling:
            sample_ratio = 0
        for batch in self.gen_lines(filenames, True, split, train, unsupervised, charmap):
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
                    with self.graph.as_default():
                        self._resync_decoder()
            else:
                lines_source, lines_sourceconf, lines_target, lines_filename = batch
                if train and self.scheduled_sampling:
                    line_schedules = np.random.uniform(0, 1, self.batch_size)
                else:
                    line_schedules = None
                # vectorize:
                encoder_input_data, decoder_input_data, decoder_output_data, decoder_output_weights = (
                    self.vectorize_lines(lines_source, lines_target, lines_sourceconf))
                # yield source/target data to keras consumer loop (fit/evaluate)
                if line_schedules is not None: # and epoch > 1:
                    # calculate greedy/beamed decoder output to yield as as decoder input
                    indexes = line_schedules < sample_ratio # respect current schedule
                    if np.count_nonzero(indexes) > 0:
                        # ensure the generator thread gets to see the same tf graph:
                        # with self.sess.as_default():
                        with self.graph.as_default():
                            decoder_input_data_sampled, _, _, _, _ = self.decode_batch_greedy(encoder_input_data)
                            # overwrite scheduled lines with data sampled from decoder instead of GT:
                            decoder_input_data.resize( # zero-fill larger time-steps (in-place)
                                decoder_input_data_sampled.shape)
                            decoder_output_data.resize( # zero-fill larger time-steps (in-place)
                                decoder_input_data_sampled.shape)
                            decoder_output_weights.resize( # zero-fill larger time-steps (in-place)
                                decoder_input_data_sampled.shape[:2])
                            indexes_condition = np.broadcast_to(indexes, # broadcast to data shape
                                                                tuple(reversed(decoder_input_data.shape))).transpose()
                            decoder_input_data = np.where(indexes_condition,
                                                          decoder_input_data_sampled,
                                                          decoder_input_data)
                if train:
                    # encoder degradation to index zero for learning character underspecification
                    rand = np.random.uniform(0, 1, self.batch_size)
                    line_length = encoder_input_data[0].shape[0]
                    rand = (line_length * rand / 0.01).astype(np.int) # effective degradation ratio
                    encoder_input_data[np.arange(self.batch_size)[rand < line_length],
                                       rand[rand < line_length], :] = np.eye(self.voc_size)[0]
                yield ([encoder_input_data, decoder_input_data],
                       decoder_output_data, decoder_output_weights)
                    
    def gen_lines(self, filenames, repeat=True, split=None, train=False, unsupervised=False, charmap={}):
        """Generate batches of lines from the given files.
        
        split...
        repeat...
        unpickle...
        normalize...
        """
        split_ratio = 0.2
        epoch = 0
        if charmap:
            charmap = str.maketrans(charmap)
        while True:
            lines_source = []
            lines_sourceconf = []
            lines_target = []
            lines_filename = []
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
                            source_text, target_text = line # already includes end-of-sequence
                            if not source_text: # empty
                                source_text, source_conf = '', []
                            elif type(source_text[0]) is tuple: # prob line
                                source_text, source_conf = map(list, zip(*source_text))
                                source_text = ''.join(source_text)
                            else: # confmat
                                source_conf = source_text
                                source_text = ''.join(chunk[0][0] if chunk else '' for chunk in source_conf)
                            # start-of-sequence will be added by vectorisation
                            # end-of-sequence already preserved by pickle format
                        elif unsupervised and '\t' not in line:
                            source_text = target_text = line
                        else:
                            source_text, target_text = line.split('\t')
                            # start-of-sequence will be added by vectorisation
                            # add end-of-sequence:
                            source_text = source_text + '\n'
                            # end-of-sequence already preserved by file iterator
                        if unsupervised:
                            target_text = source_text
                        if charmap:
                            source_text = source_text.translate(charmap)
                            target_text = target_text.translate(charmap)
                        source_text = unicodedata.normalize('NFC', source_text)
                        target_text = unicodedata.normalize('NFC', target_text)

                        if train:
                            # align source and target text line:
                            self.aligner.set_seqs(source_text, target_text)
                            if self.aligner.is_bad():
                                if epoch == 0:
                                    self.logger.debug('%s' 'ignoring bad line "%s\t%s"',
                                                      '\x1b[2K\x1b[G' if self.progbars else '',
                                                      source_text.rstrip(), target_text.rstrip())
                                continue # avoid training if OCR was too bad
                        
                        lines_source.append(source_text)
                        lines_target.append(target_text)
                        if with_confidence:
                            lines_sourceconf.append(source_conf)
                        lines_filename.append(filename)

                        if len(lines_source) == self.batch_size: # end of batch
                            yield (lines_source, lines_sourceconf if with_confidence else None,
                                   lines_target, lines_filename)
                            lines_source = []
                            lines_sourceconf = []
                            lines_target = []
                            lines_filename = []
            epoch += 1
            if repeat:
                yield False
                # bury remaining lines (partially filled batch)
            else:
                if lines_source:
                    # a partially filled batch remains
                    lines_source.extend((self.batch_size-len(lines_source))*[''])
                    lines_target.extend((self.batch_size-len(lines_target))*[''])
                    if with_confidence:
                        lines_sourceconf.extend((self.batch_size-len(lines_sourceconf))*[[]])
                    lines_filename.extend((self.batch_size-len(lines_filename))*[None])
                    yield (lines_source, lines_sourceconf if with_confidence else None,
                           lines_target, lines_filename)
                break
    
    def vectorize_lines(self, encoder_input_sequences, decoder_input_sequences, encoder_conf_sequences=None):
        '''Convert a batch of source and target sequences to arrays.
        
        Take the given (line) lists of encoder and decoder input strings,
        `encoder_input_sequences` and `decoder_input_sequences`, map them
        to indexes in the input dimension, and turn them into unit vectors,
        padding each string to the longest line using zero vectors.
        This gives numpy arrays of shape (batch_size, max_length, voc_size).
        
        When `encoder_conf_sequences` is also given, use floating point
        probability values instead of integer ones. This can come in either
        of two forms: simple lists of probabilities (of equal length as the
        strings themselves), or full confusion networks, where every line
        is a list of chunks, and each chunk is a list of alternatives, which
        is a tuple of a string and its probability. (Chunks/alternatives may
        have different length.)
        
        Special cases:
        - true zero (no index): padding for encoder and decoder (masked),
                                and start "symbol" for decoder input
        - empty character (index zero): underspecified encoder input
                                        (not allowed in decoder)
        '''
        # Note: padding and confidence indexing need Dense/dot instead of Embedding/gather.
        # Used both for training (teacher forcing) and inference (ignore decoder input/output/weights).
        max_encoder_input_length = max(map(len, encoder_input_sequences))
        max_decoder_input_length = max(map(len, decoder_input_sequences))
        assert len(encoder_input_sequences) == len(decoder_input_sequences)
        batch_size = len(encoder_input_sequences)
        with_confmat = False
        if encoder_conf_sequences:
            assert len(encoder_conf_sequences) == len(encoder_input_sequences)
            if type(encoder_conf_sequences[0][0]) is list:
                with_confmat = True
                max_encoder_input_length = max(
                    [sum(max([len(x[0]) for x in chunk]) if chunk else 0
                         for chunk in sequence)
                     for sequence in encoder_conf_sequences])
                encoder_input_sequences = encoder_conf_sequences
        encoder_input_data  = np.zeros((batch_size, max_encoder_input_length, self.voc_size),
                                       dtype=np.float32 if encoder_conf_sequences else np.uint32)
        decoder_input_data  = np.zeros((batch_size, max_decoder_input_length+1, self.voc_size),
                                       dtype=np.uint32)
        decoder_output_data = np.zeros((batch_size, max_decoder_input_length+1, self.voc_size),
                                       dtype=np.uint32)
        for i, (enc_seq, dec_seq) in enumerate(zip(encoder_input_sequences, decoder_input_sequences)):
            j = 0 # to declare scope outside loop
            if with_confmat:
                for chunk in enc_seq:
                    max_chars = max([len(x[0]) for x in chunk]) if chunk else 0
                    for chars, conf in chunk:
                        for k, char in enumerate(chars):
                            if char not in self.mapping[0]:
                                if char != GAP:
                                    self.logger.error('unmapped character "%s" at encoder input sequence %d position %d',
                                                      char, i, j+k)
                                idx = 0 # underspecification
                            else:
                                idx = self.mapping[0][char]
                            encoder_input_data[i, j+k, idx] = conf
                        # ...other k for input: padding (keep zero)
                    j += max_chars
                # ...other j for input: padding (keep zero)
            else:
                for j, char in enumerate(enc_seq):
                    if char not in self.mapping[0]:
                        if char != GAP:
                            self.logger.error('unmapped character "%s" at encoder input sequence %d', char, i)
                        idx = 0 # underspecification
                    else:
                        idx = self.mapping[0][char]
                    encoder_input_data[i, j, idx] = 1
                    if encoder_conf_sequences: # binary input with OCR confidence?
                        encoder_input_data[i, j, idx] = encoder_conf_sequences[i][j]
                # ...other j for encoder input: padding (keep zero)
            # j == 0 for decoder input: start symbol (keep zero)
            for j, char in enumerate(dec_seq):
                if char not in self.mapping[0]:
                    if char != GAP:
                        self.logger.error('unmapped character "%s" at decoder input sequence %d', char, i)
                    idx = 0
                else:
                    idx = self.mapping[0][char]
                decoder_input_data[i, j+1, idx] = 1
                # teacher forcing:
                decoder_output_data[i, j, idx] = 1
            # j == len(dec_seq) for decoder output: padding (keep zero)
            # ...other j for decoder input and output: padding (keep zero)
        
        # index of padded samples, so we can mask them
        # with the sample_weight parameter during fit() below
        decoder_output_weights = np.ones(decoder_output_data.shape[:-1], dtype=np.float32)
        decoder_output_weights[np.all(decoder_output_data == 0, axis=2)] = 0. # true zero (padding)
        #sklearn.preprocessing.normalize(decoder_output_weights, norm='l1', copy=False) # since Keras 2.3
        if self.lm_loss:
            # 2 outputs, 1 combined loss:
            decoder_output_data = [decoder_output_data, decoder_output_data]
            decoder_output_weights = [decoder_output_weights, decoder_output_weights]
        
        return encoder_input_data, decoder_input_data, decoder_output_data, decoder_output_weights
    
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
        self.encoder_decoder_model.load_weights(filename, by_name=True)
        self._resync_decoder()
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
                self._reconfigure_for_mapping()
                if config['depth'][()] == self.depth - 1:
                    was_shallow = True
            self.logger.info('Transferring model from "%s"', filename)
            load_weights_from_hdf5_group_by_name(file,
                                                 [layer.cell # LM does not have attention wrapper in top HL
                                                  if layer.name == 'decoder_lstm_%d' % self.depth
                                                  else layer
                                                  for layer in self.encoder_decoder_model.layers],
                                                 skip_mismatch=True, reshape=False)
            if was_shallow:
                self.logger.info('fixing weights from shallower model')
                for i in range(1, self.depth): # fix previous layer weights
                    self.encoder_decoder_model.get_layer(name='encoder_lstm_%d'%i).trainable = False
                    self.encoder_decoder_model.get_layer(name='decoder_lstm_%d'%i).trainable = False
                self._recompile() # necessary for trainable to take effect
            self._resync_decoder()
        self.status = 1
        
    def decode_batch_greedy(self, encoder_input_data):
        '''Predict from one batch of lines array without alternatives.
        
        Use encoder input lines array `encoder_input_data` (in a full batch)
        to produce some encoder output to attend to.
        
        Start decoder with start-of-sequence, then keep decoding until
        end-of-sequence is found or output length is way off.
        Decode by using the full output distribution as next input.
        Pass decoder initial/final states from character to character.
        
        Return a 5-tuple of the full output array (for training phase),
        output strings, output probability lists, entropies, and soft
        alignments (input-output matrices as list of list of vectors).
        '''
        
        encoder_outputs = self.encoder_model.predict_on_batch(encoder_input_data)
        encoder_output_data = encoder_outputs[0]
        states_values = encoder_outputs[1:]
        batch_size = encoder_input_data.shape[0]
        batch_length = encoder_input_data.shape[1]
        decoder_input_data = np.zeros((batch_size, 1, self.voc_size), dtype=np.uint32)
        decoder_output_data = np.zeros((batch_size, batch_length * 2, self.voc_size), dtype=np.uint32)
        decoder_output_sequences = [''] * batch_size
        decoder_output_probs = [[] for _ in range(batch_size)]
        decoder_output_scores = [0.] * batch_size
        #decoder_output_alignments = [[]] * batch_size # does not copy!!
        decoder_output_alignments = [[] for _ in range(batch_size)]
        for i in range(batch_length * 2):
            decoder_output_data[:, i] = decoder_input_data[:, -1]
            output = self.decoder_model.predict_on_batch(
                [decoder_input_data, encoder_output_data] + states_values)
            scores = output[0]
            states_values = list(output[1:])
            alignment = states_values[-1]
            indexes = np.nanargmax(scores[:, :, 1:], axis=2) # without index zero (underspecification)
            #decoder_input_data = np.eye(self.voc_size, dtype=np.uint32)[indexes+1] # unit vectors
            decoder_input_data = scores # soft/confidence input (much better)
            logscores = -np.log(scores)
            for j, idx in enumerate(indexes[:, -1] + 1):
                if decoder_output_sequences[j].endswith('\n') or not np.any(encoder_input_data[j]):
                    continue
                decoder_output_sequences[j] += self.mapping[1][idx]
                decoder_output_probs[j].append(scores[j, -1, idx])
                decoder_output_scores[j] += logscores[j, -1, idx]
                decoder_output_alignments[j].append(alignment[j])
        for j in range(batch_size):
            if decoder_output_sequences[j]:
                decoder_output_scores[j] /= len(decoder_output_sequences[j])
        # # calculate rejection scores (decoder input = encoder input):
        # decoder_input_data = np.insert(encoder_input_data, 0, 0., axis=1) # add start-of-sequence
        # decoder_rej_sequences = [''] * batch_size
        # decoder_rej_scores = [0.] * batch_size
        # output = self.decoder_model.predict_on_batch([decoder_input_data, encoder_output_data] + encoder_outputs[1:])
        # logscores = np.log(output[0])
        # for i in range(batch_length):
        #     indexes = np.nanargmax(encoder_input_data[:, i], axis=1)
        #     for j, idx in enumerate(indexes):
        #         if decoder_rej_sequences[j].endswith('\n') or not np.any(encoder_input_data[j]):
        #             continue
        #         decoder_rej_sequences[j] += self.mapping[1][int(idx)]
        #         decoder_rej_scores[j] -= logscores[j, i, idx]
        # for j in range(batch_size):
        #     if len(decoder_rej_sequences[j]) > 0:
        #         decoder_rej_scores[j] /= len(decoder_rej_sequences[j])
        #         # select rejection if better than decoder output:
        #         if decoder_rej_scores[j] < decoder_output_scores[j]:
        #             decoder_output_sequences[j] = decoder_rej_sequences[j]
        #             decoder_output_scores[j] = decoder_rej_scores[j]
        return (decoder_output_data,
                decoder_output_sequences, decoder_output_probs,
                decoder_output_scores, decoder_output_alignments)
    
    def decode_sequence_greedy(self, source_seq=None, encoder_outputs=None):
        '''Predict from one line vector without alternatives.
        
        Use encoder input line vector `source_seq` (in a batch of size 1)
        to produce some encoder output to attend to.
        If `encoder_outputs` is given, then bypass that step.
        
        Start decoder with start-of-sequence, then keep decoding until
        end-of-sequence is found or output length is way off.
        Decode by using the full output distribution as next input.
        Pass decoder initial/final states from character to character.
        
        Return a 4-tuple of output string, output probabilities, entropy,
        and soft alignment (input-output matrix as list of vectors).
        '''
        
        # Encode the source as state vectors.
        if encoder_outputs is None:
            encoder_outputs = self.encoder_model.predict_on_batch(np.expand_dims(source_seq, axis=0))
        attended_seq = encoder_outputs[0]
        states_values = encoder_outputs[1:]
        
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.voc_size), dtype=np.uint32)
        # The first character (start symbol) stays empty.
        
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        decoded_text = ''
        decoded_probs = []
        decoded_score = 0
        alignments = []
        for i in range(attended_seq.shape[1] * 2):
            output = self.decoder_model.predict_on_batch([target_seq, attended_seq] + states_values)
            scores = output[0]
            if self.lm_predict:
                states = output[2:]
            else:
                states = output[1:]
            
            # Sample a token:
            idx = np.nanargmax(scores[0, -1, :])
            prob = scores[0, -1, idx]
            score = -np.log(prob)
            char = self.mapping[1][idx]
            if char == '': # underspecification
                scores[0, -1, idx] = np.nan
                idx = np.nanargmax(scores[0, -1, :])
                prob = scores[0, -1, idx]
                score = -np.log(prob)
                char = self.mapping[1][idx]
            decoded_text += char
            decoded_probs.append(prob)
            decoded_score += score
            alignments.append(states[-1][0])
            # Exit condition: end-of-sequence character.
            if char == '\n':
                break
            
            # Update the target sequence (of length 1):
            #target_seq = np.eye(self.voc_size, dtype=np.uint32)[[[[idx]]]]
            target_seq = scores # soft/confidence input (better)
            # Update states:
            states_values = list(states)
        
        return (decoded_text, decoded_probs,
                decoded_score / len(decoded_text), alignments)
    
    def decode_sequence_beam(self, source_seq=None, encoder_outputs=None):
        '''Predict from one line vector with alternatives.
        
        Use encoder input line vector `source_seq` (in a batch of size 1)
        to produce some encoder output to attend to.
        If `encoder_outputs` is given, then bypass that step.
        
        Start decoder with start-of-sequence, then keep decoding until
        end-of-sequence is found or output length is way off, repeatedly.
        Decode by using the best predicted output characters and several next-best
        alternatives (up to some degradation threshold) as next input.
        Follow-up on the N best overall candidates (estimated by accumulated
        score, normalized by length and prospective cost), i.e. do A*-like
        breadth-first search, with N equal `batch_size`.
        Pass decoder initial/final states from character to character,
        for each candidate respectively.
        Reserve 1 candidate per iteration for running through `source_seq`
        (as a rejection fallback) to ensure that path does not fall off the
        beam and at least one solution can be found within the search limits.
        
        For each solution, yield a 4-tuple of output string, output probabilities,
        entropy, and soft alignment (input-output matrix as list of vectors).
        '''
        from bisect import insort_left
        
        # Encode the source as state vectors.
        if encoder_outputs is None:
            encoder_outputs = self.encoder_model.predict_on_batch(np.expand_dims(source_seq, axis=0))
        attended_seq = encoder_outputs[0] # constant
        attended_len = attended_seq.shape[1]
        states_values = encoder_outputs[1:]
        
        # Start with an empty beam (no input, only state):
        next_beam = [Node(state=states_values,
                          value='', scores=np.zeros(self.voc_size),
                          prob=[], cost=0.0,
                          alignment=[],
                          length0=attended_len,
                          cost0=3.0)] # quite pessimistic
        final_beam = []
        # how many batches (i.e. char hypotheses) will be processed per line at maximum?
        max_batches = attended_len * 2 # (usually) safe limit
        for l in range(max_batches):
            beam = []
            while next_beam:
                node = next_beam.pop()
                if node.value == '\n': # end-of-sequence symbol?
                    insort_left(final_beam, node)
                    # self.logger.debug('%02d found new solution %.2f/"%s"',
                    #                   l, node.pro_cost(), str(node).strip('\n'))
                else: # normal step
                    beam.append(node)
                    if node.length > 1.5 * attended_len:
                        self.logger.warning('found overlong hypothesis "%s" in "%s"',
                                            str(node),
                                            ''.join(self.mapping[1][np.nanargmax(step)] for step in source_seq))
                    # self.logger.debug('%02d new hypothesis %.2f/"%s"',
                    #                   l, node.pro_cost(), str(node).strip('\n'))
                if len(beam) >= self.batch_size:
                    break # enough for one batch
            if not beam:
                break # will yield StopIteration unless we have some results already
            if (len(final_beam) > self.beam_width_out and
                final_beam[-1].pro_cost() > beam[0].pro_cost()):
                break # it is unlikely that later iterations will find better top n results
            
            # use fringe leaves as minibatch, but with only 1 timestep
            target_seq = np.expand_dims(
                np.vstack([node.scores for node in beam]),
                axis=1) # add time dimension
            states_val = [np.vstack([node.state[layer] for node in beam])
                          for layer in range(len(beam[0].state))] # stack layers across batch
            output = self.decoder_model.predict_on_batch(
                [target_seq, attended_seq] + states_val)
            scores_output = output[0][:, -1] # only last timestep
            if self.lm_predict:
                lmscores_output = output[1][:, -1]
                states_output = list(output[2:])
            else:
                states_output = list(output[1:]) # from (layers) tuple
            for i, node in enumerate(beam): # iterate over batch (1st dim)
                # unstack layers for current sample:
                states = [layer[i:i+1] for layer in states_output]
                scores = scores_output[i]
                #
                # estimate current alignment target:
                alignment = states[-1][0]
                misalignment = 0.0
                if node.length > 1:
                    prev_alignment = node.alignment
                    prev_source_pos = np.matmul(prev_alignment, np.arange(attended_len))
                    source_pos = np.matmul(alignment, np.arange(attended_len))
                    misalignment = np.abs(source_pos - prev_source_pos - 1)
                    if np.max(prev_alignment) == 1.0:
                        # previous choice was rejection
                        source_pos = int(prev_source_pos) + 1
                    else:
                        source_pos = int(source_pos.round())
                else:
                    source_pos = 0
                #
                # add fallback/rejection candidates regardless of beam threshold:
                source_scores = source_seq[source_pos]
                if (self.rejection_threshold
                    and (misalignment < 0.1 or np.max(node.alignment) == 1.0)
                    and np.any(source_scores)):
                    rej_idx = np.nanargmax(source_scores)
                    # use a fixed minimum probability
                    if scores[rej_idx] < self.rejection_threshold:
                        #scores *= self.rejection_threshold - scores[rej_idx] # renormalize
                        scores[rej_idx] = self.rejection_threshold # overwrite
                    # self.logger.debug('%s: rej=%s (%.2f)', str(node),
                    #                   self.mapping[1][rej_idx], scores[rej_idx])
                else:
                    rej_idx = None
                # 
                # determine beam width from beam threshold to add normal candidates:
                scores_order = np.argsort(scores) # still in reverse order (worst first)
                highest = scores[scores_order[-1]]
                beampos = self.voc_size - np.searchsorted(
                    scores[scores_order],
                    #highest - self.beam_threshold_in) # variable beam width (absolute)
                    highest * self.beam_threshold_in) # variable beam width (relative)
                #beampos = self.beam_width_in # fixed beam width
                beampos = min(beampos, self.beam_width_in) # mixed beam width
                pos = 0
                #
                # follow up on best predictions, in true order (best first):
                for idx in reversed(scores_order):
                    pos += 1
                    score = scores[idx]
                    logscore = -np.log(score)
                    if self.lm_predict:
                        # use probability from LM instead of decoder for beam ratings
                        logscore = -np.log(lmscores_output[i][idx])
                    alignment1 = alignment
                    if idx == rej_idx:
                        # self.logger.debug('adding rejection candidate "%s" [%.2f]',
                        #                   self.mapping[1][rej_idx], logscore)
                        alignment1 = np.eye(attended_len)[source_pos]
                        rej_idx = None
                    elif pos > beampos:
                        if rej_idx: # not yet in beam
                            continue # search for rejection candidate
                        else:
                            break # ignore further alternatives
                    #
                    # decode into string:
                    value = self.mapping[1][idx]
                    if (np.isnan(logscore) or
                        value == ''): # underspecification
                        continue # ignore this alternative
                    #
                    # add new hypothesis to the beam:
                    # for decoder feedback, use a compromise between
                    #  - raw predictions (used in greedy decoder,
                    #    still informative of ambiguity), and
                    #  - argmax unit vectors (allowing alternatives,
                    #    but introducing label bias)
                    scores1 = np.copy(scores)
                    # already slightly better than unit vectors:
                    # scores1 *= scores[idx] / highest
                    # scores1[idx] = scores[idx] # keep
                    # only disable maxima iteratively:
                    scores[idx] = 0
                    new_node = Node(parent=node, state=states,
                                    value=value, scores=scores1,
                                    prob=score, cost=logscore,
                                    alignment=alignment1)
                    # self.logger.debug('pro_cost: %3.3f, cum_cost: %3.1f, "%s"',
                    #                   new_node.pro_cost(),
                    #                   new_node.cum_cost,
                    #                   str(new_node).strip('\n'))
                    insort_left(next_beam, new_node)
            # sanitize overall beam size:
            if len(next_beam) > max_batches * self.batch_size: # more than can ever be processed within limits?
                next_beam = next_beam[-max_batches*self.batch_size:] # to save memory, keep only best
        # after max_batches, we still have active hypotheses but to few inactive?
        if next_beam and len(final_beam) < self.beam_width_out:
            self.logger.warning('max_batches %d is not enough for beam_width_out %d: got only %d, still %d left for: "%s"',
                                max_batches, self.beam_width_out, len(final_beam), len(next_beam),
                                ''.join(self.mapping[1][np.nanargmax(step)] for step in source_seq))
        while final_beam:
            node = final_beam.pop()
            nodes = node.to_sequence()[1:]
            yield (''.join(n.value for n in nodes),
                   [n.prob for n in nodes],
                   node.cum_cost / (node.length - 1),
                   [n.alignment for n in nodes])

class Node(object):
    """One hypothesis in the character beam (trie)"""
    def __init__(self, state, value, scores, cost, parent=None, prob=1.0, alignment=None, length0=None, cost0=None):
        super(Node, self).__init__()
        self._sequence = None
        self.value = value # character
        self.parent = parent # parent Node, None for root
        self.state = state # recurrent layer hidden state
        self.cum_cost = parent.cum_cost + cost if parent else cost # e.g. -log(p) of sequence up to current node (including)
        # length of 
        self.length = 1 if parent is None else parent.length + 1
        # length of source sequence (for A* prospective cost estimation)
        self.length0 = length0 or (parent.length0 if parent else 1)
        # additional (average) per-node costs
        self.cost0 = cost0 or (parent.cost0 if parent else 0)
        # urgency? (l/max_batches)...
        # probability
        self.prob = prob
        self.scores = scores
        if alignment is None:
            self.alignment = parent.alignment if parent else []
        else:
            self.alignment = alignment
    
    def to_sequence(self):
        # Return sequence of nodes from root to current node.
        if not self._sequence:
            self._sequence = []
            current_node = self
            while current_node:
                self._sequence.insert(0, current_node)
                current_node = current_node.parent
        return self._sequence
    
    def __str__(self):
        return ''.join(n.value for n in self.to_sequence()[1:])
    
    # for sort order, use cumulative costs relative to length
    # (in order to get a fair comparison across different lengths,
    #  and hence, breadth-first search), and use inverse order
    # (so the faster bisect() and pop() can be used)
    # [must be pessimistic estimation of final cum_cost]
    def pro_cost(self):
        # v0.1.0:
        #return - (self.cum_cost + 0.5 * math.fabs(self.length - self.length0)) / self.length
        # v0.1.1:
        #return - (self.cum_cost + self.cum_cost/self.length * max(0, self.length0 - self.length)) / self.length
        # v0.1.2:
        #return - (self.cum_cost + (28 + self.cost0) * self.length0 * np.abs(1 - np.sqrt(self.length / self.length0)))
        return - (self.cum_cost + self.cost0 * np.abs(self.length - self.length0))
    
    def __lt__(self, other):
        return self.pro_cost() < other.pro_cost()
    def __le__(self, other):
        return self.pro_cost() <= other.pro_cost()
    def __eq__(self, other):
        return self.pro_cost() == other.pro_cost()
    def __ne__(self, other):
        return self.pro_cost() != other.pro_cost()
    def __gt__(self, other):
        return self.pro_cost() > other.pro_cost()
    def __ge__(self, other):
        return self.pro_cost() >= other.pro_cost()
