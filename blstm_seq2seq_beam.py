'''Sequence to sequence example in Keras (byte-level).

Adapted from examples/lstm_seq2seq.py (tutorial by Francois Chollet 
"A ten-minute introduction...") with changes as follows:

- extended for use of full (large) dataset: training uses generator
  function (but not fit_generator because we want to have an automatic
  validation split for early stopping in each increment)
- use early stopping to prevent overfitting
- show results not only on training set, but validation set as well
- preprocessing utilising Keras API: pad_sequences, to_categorical etc
  (tokeniser not even necessary for byte strings)
- add end-of-sequence symbol to encoder input in training and inference
- based on byte level instead of character level
- padding needs to be taken into account by loss function: mask
  samples using sample_weight
- added preprocessing function for conveniently testing encoder
- added bidirectional and 3 more unidirectional LSTM layers
- added beam search decoding (which enforces utf-8 via incremental decoder)
- added command line arguments

Features still (very much) wanting of implementation:

- true evaluation (character/word error rates)
- attention
- systematic hyperparameter treatment (deviations from Sutskever
  should be founded on empirical analysis): 
  HL width and depth, optimiser choice (RMSprop/SGD) and parameters,
  gradient clipping, decay and rate control, initialisers

Command line interface:
$ python [-i] blstm_seq2seq_beam.py [model_filename [depth [beam_width]]]
(If model_filename already exists, it will be loaded and training will be skipped.
 Use -i to enter REPL afterwards.)

Below is description of original script:

This script demonstrates how to implement a basic character-level
sequence-to-sequence model. We apply it to translating
short English sentences into short French sentences,
character-by-character. Note that it is fairly unusual to
do character-level machine translation, as word-level
models are more common in this domain.

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

# Data download

English to French sentence pairs.
http://www.manythings.org/anki/fra-eng.zip

Lots of neat sentence pairs datasets can be found at:
http://www.manythings.org/anki/

# References

- Sequence to Sequence Learning with Neural Networks
    https://arxiv.org/abs/1409.3215
- Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    https://arxiv.org/abs/1406.1078
'''
from __future__ import print_function
from os.path import isfile
import sys
import codecs

# these should all be wrapped in functions
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Bidirectional, Dense, concatenate
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop, SGD
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

batch_size = 64  # Batch size for training.
epochs = 100  # Maximum number of epochs to train for.
latent_dim = 320  # Latent dimensionality of the encoding space.
#num_trainsamples = 10000  # Number of samples to train on.
data_path = 'fra-eng/fra.txt' # Path to the data txt file on disk.
model_filename = 's2s.3lstm+blstm320.large.h5'
if len(sys.argv) > 1:
    model_filename = sys.argv[1]
depth = 4 # number of encoder/decoder layers stacked above each other (only 1st layer will be BLSTM)
if len(sys.argv) > 2:
    depth = int(sys.argv[2])
beam_width = 4 # keep track of how many alternative sequences during decode_sequence_beam()?
if len(sys.argv) > 3:
    beam_width = int(sys.argv[3])
beam_width_out = 4 # up to how many results can be drawn from generator decode_sequence_beam()?

# effort for CPU-only (4-core Ryzen 5 3.2 GHz) training (roughly constant throughout training set despite its increasing sequence lengths because generator draws chunks by storage size, not number of lines):
# * depth 1 LSTM on (selected) chars with 256 hidden nodes ~30s per epoch (does not converge on last sequences)
# * depth 1 LSTM on bytes with 256 hidden nodes ~60s per epoch (does not converge on last sequences)
# * depth 1 LSTM on bytes with 512 hidden nodes ~120-80s per epoch (does not converge on last sequences)
# * depth 1 BLSTM on bytes with 320 hidden nodes: ~160-100-130s per epoch
# * depth 4 with BLSTM ground layer on bytes with 320 hidden nodes per layer: ~340s per epoch
#
# By the course of training and validation loss with each incremental batch
# (with increasing sequence lengths), and by sporadic testing, it seems
# that this data is too small to make even a large network converge.
# Validation loss keeps declining until sequences become about 100 characters long.
# From there early stopping will already show degradation at the very first epoch,
# and the longer training sequences in the batches get, the more model performance
# degrades even on short sequences.
# Since this 4-layer network with bidirectional base already has 5M parameters,
# it should be large enough for the problem, despite the large distance between encoder
# start and decoder stop. So most likely the 11MB dataset is too small to be representative
# for character-level language modelling.

# --->8--- from Keras FAQ: How can I obtain reproducible results using Keras during development?

# import tensorflow as tf
# import random as rn

# # The below is necessary in Python 3.2.3 onwards to
# # have reproducible behavior for certain hash-based operations.
# # See these references for further details:
# # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# # https://github.com/keras-team/keras/issues/2280#issuecomment-306959926

# import os
# os.environ['PYTHONHASHSEED'] = '0'

# # The below is necessary for starting Numpy generated random numbers
# # in a well-defined initial state.

# np.random.seed(42)

# # The below is necessary for starting core Python generated random numbers
# # in a well-defined state.

# rn.seed(12345)

# # Force TensorFlow to use single thread.
# # Multiple threads are a potential source of
# # non-reproducible results.
# # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

# from keras import backend as K

# # The below tf.set_random_seed() will make random number generation
# # in the TensorFlow backend have a well-defined initial state.
# # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

# tf.set_random_seed(1234)

# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)

# --->8---


# Vectorize the data.

# for char in source_text:
#     if char not in source_characters:
#         source_characters.add(char)
# for char in target_text:
#     if char not in target_characters:
#         target_characters.add(char)
#source_tokenizer = Tokenizer(char_level=True)
#source_tokenizer.fit_on_texts(source_texts)
#source_token_index = source_tokenizer.word_index
#num_encoder_tokens = len(source_token_index) # tokenizer index starts with 1
num_encoder_tokens = 255 # now utf8 bytes (non-nul, but including tab for start and newline for stop)

#target_tokenizer = Tokenizer(char_level=True)
#target_tokenizer.fit_on_texts(target_texts)
#target_token_index = target_tokenizer.word_index
#num_decoder_tokens = len(target_token_index) # tokenizer index starts with 1
num_decoder_tokens = 255 # now utf8 bytes

# source_characters = sorted(list(source_characters))
# target_characters = sorted(list(target_characters))
# num_encoder_tokens = len(source_characters)
# num_decoder_tokens = len(target_characters)
# source_token_index = dict(
#     [(char, i) for i, char in enumerate(source_characters)])
# target_token_index = dict(
#     [(char, i) for i, char in enumerate(target_characters)])

# encoder_input_data = np.zeros(
#     (num_samples, max_encoder_seq_length, num_encoder_tokens),
#     dtype='float32')
# decoder_input_data = np.zeros(
#     (num_samples, max_decoder_seq_length, num_decoder_tokens),
#     dtype='float32')
# decoder_output_data = np.zeros(
#     (num_samples, max_decoder_seq_length, num_decoder_tokens),
#     dtype='float32')

# for i, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
#     for t, char in enumerate(source_text):
#         encoder_input_data[i, t, source_token_index[char]] = 1.
#     for t, char in enumerate(target_text):
#         # decoder_output_data is ahead of decoder_input_data by one timestep
#         decoder_input_data[i, t, target_token_index[char]] = 1.
#         if t > 0:
#             # decoder_output_data will be ahead by one timestep
#             # and will not include the start character.
#             decoder_output_data[i, t - 1, target_token_index[char]] = 1.

def gen_data(filename):
    with open(filename, 'rb') as file:
        while True:
            lines = file.readlines(500000) # no more than 512 kB at once
            if not lines and file.tell() > 0:
                break
            #     file.seek(0) # make sure the generator wraps around
            #     gen_data(file)
            source_texts = []
            target_texts = []
            for line in lines:
                if not line: # empty
                    continue
                source_text, target_text = line.split(b'\t')
                source_text = source_text + b'\n' # add end-of-sequence
                target_text = b'\t' + target_text # add start-of-sequence (readlines already keeps newline as end-of-sequence)
                source_texts.append(source_text)
                target_texts.append(target_text)
            max_encoder_seq_length = max([len(txt) for txt in source_texts])
            max_decoder_seq_length = max([len(txt) for txt in target_texts])
            num_samples = len(source_texts)
            print('Number of samples:', num_samples)
            #print('Number of unique source tokens:', num_encoder_tokens)
            #print('Number of unique target tokens:', num_decoder_tokens)
            print('Max sequence length for sources:', max_encoder_seq_length)
            print('Max sequence length for targets:', max_decoder_seq_length)
            
            #encoder_input_sequences = source_tokenizer.texts_to_sequences(source_texts)
            #decoder_input_sequences = target_tokenizer.texts_to_sequences(target_texts)
            encoder_input_sequences = list(map(bytearray,source_texts))
            decoder_input_sequences = list(map(bytearray,target_texts))
            encoder_input_sequences = pad_sequences(encoder_input_sequences, maxlen=max_encoder_seq_length, padding='post')
            decoder_input_sequences = pad_sequences(decoder_input_sequences, maxlen=max_decoder_seq_length, padding='post')
            encoder_input_data = to_categorical(encoder_input_sequences, num_classes=num_encoder_tokens+1)
            decoder_input_data = to_categorical(decoder_input_sequences, num_classes=num_decoder_tokens+1)
            encoder_input_data = encoder_input_data[:,:,1:] # remove separate dimension for zero/padding
            decoder_input_data = decoder_input_data[:,:,1:] # remove separate dimension for zero/padding
            # teacher forcing:
            decoder_output_data = np.roll(decoder_input_data,-1,axis=1) # output data will be ahead by 1 timestep
            decoder_output_data[:,-1,:] = np.zeros(num_decoder_tokens) # delete+pad start token rolled in at the other end
            
            # index of padded samples, so we can mask them with the sample_weight parameter during fit() below
            decoder_output_weights = np.ones(decoder_input_sequences.shape, dtype=np.float32)
            decoder_output_weights[np.all(decoder_output_data == 0, axis=2)] = 0.
            
            # shuffle within chunk so validation split does not get the longest sequences only
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            encoder_input_data = encoder_input_data[indices]
            decoder_input_data = decoder_input_data[indices]
            decoder_output_data = decoder_output_data[indices]
            decoder_output_weights = decoder_output_weights[indices]
            
            yield ([encoder_input_data, decoder_input_data], decoder_output_data, decoder_output_weights)


### Define training mode model

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
if depth < 2:
    encoder = Bidirectional(LSTM(latent_dim, return_state=True)) # dropout/recurrent_dropout does not seem to help (at least for small unidirectional encoder), go_backwards helps for unidirectional encoder with ~0.1 smaller loss on validation set (and not any slower) unless UTF-8 byte strings are used directly
    encoder_outputs, fw_state_h, fw_state_c, bw_state_h, bw_state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    state_h = concatenate([fw_state_h, bw_state_h])
    state_c = concatenate([fw_state_c, bw_state_c])
    encoder_states = [state_h, state_c]
else:
    encoder_outputs, fw_state_h, fw_state_c, bw_state_h, bw_state_c = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True))(encoder_inputs)
    state_h = concatenate([fw_state_h, bw_state_h])
    state_c = concatenate([fw_state_c, bw_state_c])
    encoder_states = [state_h, state_c]
    for n in range(1,depth):
        encoder_outputs, state_h, state_c = LSTM(latent_dim, return_sequences=(n < depth-1), return_state=True)(encoder_outputs)
        encoder_states = encoder_states + [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim*2, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states[0:2])
if depth >= 2:
    decoders_lstm = [decoder_lstm]
    for n in range(1,depth):
        decoders_lstm = decoders_lstm + [LSTM(latent_dim, return_sequences=True, return_state=True)]
        decoder_outputs, _, _ = decoders_lstm[n](decoder_outputs,
                                                 initial_state=encoder_states[2*n:2*n+2])
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_output_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


# Define inference mode model
# Here's the drill:
# 1) encode source and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat from 2 with the current target token and current states

# encoder can be re-used unchanged (but with result states as output)
encoder_model = Model(encoder_inputs, encoder_states)
# decoder must be re-defined
decoder_state_input_h = Input(shape=(latent_dim*2,))
decoder_state_input_c = Input(shape=(latent_dim*2,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
if depth >= 2:
    for n in range(1,depth):
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = decoder_states_inputs + [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoders_lstm[n](
            decoder_outputs, initial_state=decoder_states_inputs[2*n:2*n+2])
        decoder_states = decoder_states + [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
    
# Reverse-lookup token index to decode sequences back to
# something readable.
#reverse_source_char_index = dict(
#    (i, char) for char, i in source_token_index.items())
#reverse_target_char_index = dict(
#    (i, char) for char, i in target_token_index.items())

def decode_sequence_beam(source_seq):
    class Node(object):
        def __init__(self, parent, state, value, cost, extras):
            super(Node, self).__init__()
            self.value = value # dimension index = byte - 1
            self.parent = parent # parent Node, None for root
            self.state = state # recurrent layer hidden state
            self.cum_cost = parent.cum_cost + cost if parent else cost # e.g. -log(p) of sequence up to current node (including)
            self.length = 1 if parent is None else parent.length + 1
            self.extras = extras
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
            return [s.value for s in self.to_sequence()]
    
    decoder = codecs.getincrementaldecoder('utf8')()
    decoder.decode(b'\t')
    next_fringe = [Node(parent=None,
                        state=encoder_model.predict(np.expand_dims(source_seq, axis=0)), # layer list of state arrays
                        value=b'\t'[0]-1, # start symbol index
                        cost=0.0,
                        extras=decoder.getstate())]
    hypotheses = []
    
    # generator will raise StopIteration if hypotheses is still empty after loop
    for l in range(50):
        # try:
        #     next(n for n in next_fringe if all(np.array_equal(x,y) for x,y in zip(nonbeam_states[:l+1], [s.state for s in n.to_sequence()])))
        # except StopIteration:
        #     print('greedy result falls off beam search at pos', l, nonbeam_seq[:l+1])
        fringe = []
        for n in next_fringe:
            if n.value == b'\n'[0]-1: # end symbol index
                hypotheses.append(n)
                #print('found new solution', bytes([i+1 for i in n.to_sequence_of_values()]).decode("utf-8", "ignore"))
            else:
                fringe.append(n)
                #print('added new hypothesis', bytes([i+1 for i in n.to_sequence_of_values()]).decode("utf-8", "ignore"))
        if not fringe or len(hypotheses) >= beam_width_out:
            break
        
        # use fringe leaves as minibatch, but with only 1 timestep
        target_seq = np.expand_dims(np.eye(256, dtype=np.float32)[[n.value+1 for n in fringe],1:], axis=1) # add time dimension
        states_val = [np.vstack([n.state[layer] for n in fringe]) for layer in range(len(fringe[0].state))] # stack layers across batch
        output = decoder_model.predict([target_seq] + states_val)
        scores_output = output[0][:,-1] # only last timestep
        scores_output_best = np.argsort(scores_output, axis=1)[:,-beam_width:] # still in reverse sort order
        states_output = list(output[1:]) # from (layers) tuple
        
        next_fringe = []
        for i, n in enumerate(fringe): # iterate over batch (1st dim)
            scores = scores_output[i,:]
            scores_best = scores_output_best[i,:]
            states = [layer[i:i+1] for layer in states_output] # unstack layers for current sample
            logscores = -np.log(scores[scores_best])
            for best, logscore in zip(scores_best, logscores): # follow up on beam_width best predictions
                decoder.setstate(n.extras)
                try:
                    decoder.decode(bytes([best+1]))
                    n_new = Node(parent=n, state=states, value=best, cost=logscore, extras=decoder.getstate())
                    next_fringe.append(n_new)
                except UnicodeDecodeError:
                    pass # ignore this alternative
        next_fringe = sorted(next_fringe, key=lambda n: n.cum_cost)[:beam_width]
    
    hypotheses.sort(key=lambda n: n.cum_cost)
    for hypothesis in hypotheses[:beam_width_out]:
        indices = hypothesis.to_sequence_of_values()
        byteseq = bytes([i+1 for i in indices])
        yield byteseq

def decode_sequence_greedy(source_seq):
    # Encode the source as state vectors.
    states_value = encoder_model.predict(np.expand_dims(source_seq, axis=0))
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    #target_seq[0, 0, target_token_index['\t']-1] = 1.
    target_seq[0, 0, b'\t'[0]-1] = 1.
    
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = b''
    while not stop_condition:
        output = decoder_model.predict(
            [target_seq] + states_value)
        output_tokens = output[0]
        output_states = output[1:]
        
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        #sampled_char = reverse_target_char_index[sampled_token_index+1]
        sampled_char = bytes([sampled_token_index+1])
        decoded_sentence += sampled_char
        
        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == b'\n' or
            len(decoded_sentence) > 400):
            stop_condition = True
        
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        # feeding back the softmax vector directly vastly deteriorates predictions (because only identity vectors are presented during training)
        # but we should add beam search (keep n best overall paths, add k best predictions here)
        
        # Update states
        states_value = list(output_states)
    
    return decoded_sentence


if isfile(model_filename):
    print('Loading model', model_filename)
    model = load_model(model_filename)
else:
    # Run training
    early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='min')
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  sample_weight_mode='temporal') # sample_weight slows down slightly (20%)

    # lines in this file are sorted by sequence length!
    # instead of fit_generator:
    for (input, output, sample_weight) in gen_data(data_path):
        model.fit(input, output,
                  batch_size=batch_size,
                  epochs=epochs,
                  callbacks=[early_stopping],
                  sample_weight=sample_weight,
                  validation_split=0.2)

        n = input[0].shape[0]
        for i in list(range(0,3))+list(range(n-3,n)):
            # Take one sequence (part of the training/validation set) and try decoding it
            source_seq = input[0][i]
            decoded_sentence = decode_sequence_greedy(source_seq)
            beamdecoded_sentence = next(decode_sequence_beam(source_seq)) # query only 1-best
            print('Source text from', 'training:' if i<n*0.8 else 'test:', (input[0][i].nonzero()[1]+1).astype(np.uint8).tobytes().decode("utf-8", "strict"))
            print('Target text from', 'training:' if i<n*0.8 else 'test:', (input[1][i].nonzero()[1]+1).astype(np.uint8).tobytes().decode("utf-8", "strict"))
            print('Target prediction (greedy):', decoded_sentence.decode("utf-8", "ignore"))
            print('Target prediction (beamed):', beamdecoded_sentence.decode("utf-8", "strict"))            

    # Save model
    print('Saving model', model_filename)
    model.save(model_filename) # FIXME: when this throws a UserWarning that 'initial_state_ is a non-serializable keyword argument for the the tensorflow tensor, then the model will be useless -- it will load successfully but produce wrong results

#with open(other_path, 'rb') as f:
#    loss = model.evaluate_generator(gen_data(f), workers=4, verbose=1)
#print('Test loss:', loss)


def encode_text(source_text):
    # source_sequence = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    # for t, char in enumerate(source_text):
    #     source_sequence[0, t, source_token_index[char]] = 1
    # return source_sequence
    #return to_categorical(pad_sequences(source_tokenizer.texts_to_sequences([source_text]), maxlen=max_encoder_seq_length, padding='pre'), num_classes=num_encoder_tokens+1)[:,:,1:] # remove separate dimension for zero/padding
    #return to_categorical(pad_sequences([list(map(bytearray,source_text))], maxlen=max_encoder_seq_length, padding='pre'), num_classes=num_encoder_tokens+1)[:,:,1:] # remove separate dimension for zero/padding
    #return to_categorical(list(map(bytearray,[source_text + b'\n'])), num_classes=num_encoder_tokens+1)[:,:,1:]
    return np.eye(256, dtype=np.float32)[bytearray(source_text+b'\n'),1:]

print("usage example:\n# for reading in decode_sequence_beam(encode_text(b'hello world!')):\n#     print(reading.decode('utf-8'))")
