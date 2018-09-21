'''Sequence to sequence translator for noisy channel error correction

Command line interface:
$ python [-i] blstm_seq2seq_beam.py [model_filename [depth [beam_width]]]
(If model_filename already exists, it will be loaded and training will be skipped.
 Use -i to enter REPL afterwards.)
'''

from __future__ import print_function
from os.path import isfile
from os import environ, rename
import sys
import codecs
import numpy as np
# these should all be wrapped in functions:
import pickle, click
from editdistance import eval as edit_eval
from alignment.sequence import Sequence
sys.modules['alignment.sequence'].GAP_ELEMENT = 0 # override default
#GAP_ELEMENT = b'\x00' # see windowing from alignment in gen_data
from alignment.sequence import GAP_ELEMENT
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, StrictGlobalSequenceAligner

# load pythonrc even with -i
if 'PYTHONSTARTUP' in environ:
    exec(open(environ['PYTHONSTARTUP']).read())

# installing CUDA, cuDNN, Tensorflow with cuDNN support:
# $ sudo add-apt-repository ppa:graphics-drivers/ppa
# $ sudo apt-get update
# $ sudo apt-get install nvidia-driver nvidia-cuda-toolkit libcupti-dev
# $ git clone https://github.com/NVIDIA/cuda-samples.git && cd cuda-samples
# nvcc does not work with v(gcc) > 5, and deb installation path has standard prefix:
# $ sudo apt-get install gcc-5 g++-5
# $ make CUDA_PATH=/usr HOST_COMPILER=g++-5 EXTRA_NVCCFLAGS="-L /usr/lib/x86_64-linux-gnu"
# $ ./bin/x86_64/linux/release/deviceQuery
#
# ... https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installcuda
# Ubuntu 18.04 is still not supported, but we can use the 16.04 deb files libcudnn7{,-dev,-doc} 
# so register as Nvidia developer, download cuDNN (https://developer.nvidia.com/rdp/cudnn-download)
# important: cuDNN depends on specific CUDA toolkit versions (without explicit dependency)!
# $ sudo dpkg -i libcudnn7-dev_7.1.4.18-1+cuda9.0_amd64.deb
# $ sudo dpkg -i libcudnn7-dev-dev_7.1.4.18-1+cuda9.0_amd64.deb
# $ sudo dpkg -i libcudnn7-doc-dev_7.1.4.18-1+cuda9.0_amd64.deb
# $ cp -r /usr/src/cudnn_samples_v7/mnistCUDNN/ . && cd mnistCUDNN
# $ make CUDA_PATH=/usr HOST_COMPILER=g++-5 EXTRA_NVCCFLAGS="-L /usr/lib/x86_64-linux-gnu"
# $ ./mnistCUDNN
#
# Tensorflow-GPU (gemäß Anleitung zum Bauen aus den Quellen mit CUDA, https://www.tensorflow.org/install/install_sources,
# aber die dortigen Angaben über LD_LIBRARY_PATH und dergl. müssen hier ignoriert werden):
# ...
# $ cd tensorflow; git checkout r1.8
# $ ./configure
# ... muß einmal für /usr/bin/python und einmal für /usr/bin/python3 gemacht werden;
#     bei CUDA y antworten, Version 9.1 und Pfad /usr eingeben (werden falsch geraten),
#     bei cuDNN 7 ebenfalls, Version 7.1 und Pfad /usr,
#     Compute-capability 6.1 (für Quadro P1000 gemäß https://developer.nvidia.com/cuda-gpus)
#     (wird ebenfalls falsch geraten)
#     Als C-Compiler /usr/bin/x86_64-linux-gnu-gcc-5 (wird falsch geraten), da >5 nicht funktioniert
#     (aber >5.4 funktioniert auch schon nicht, s.u.)
# ... damit /usr überhaupt funktioniert, muß man third_party/gpus/cuda_configure.bzl patchen!
# ... Außerdem scheint es einen Bug in GCC 5.5 zu geben (bei den intrinsischen Funktionen für AVX512-Befehlssatz):
#     https://github.com/tensorflow/tensorflow/issues/10220#issuecomment-352110064
#     Mit diesem Workaround läuft folgender Build dann durch:
# $ bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
# $ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# $ . env/bin/activate
# $ pip install /tmp/tensorflow_pkg/tensorflow-1.8.0-cp27-cp27mu-linux_x86_64.whl
# (oder pip install --ignore-installed --upgrade ...)
# $ pip install pycudnn
# $ deactivate
# ...
# $ . env3/bin/activate
# $ pip install /tmp/tensorflow_pkg/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl
# $ pip install pycudnn
# $ deactivate
#

# CuDNNLSTM // effort for GPU (Quadro P1000 5*128 CUDA cores 1.5 GHz 4 GB CUDA 9.1 CC 6.1):
# * depth 4 with BLSTM ground layer on bytes with 320 hidden nodes per layer: ~39s per epoch
# LSTM // effort for CPU-only (4-core Ryzen 5 3.2 GHz) training (roughly constant throughout training set despite its increasing sequence lengths because generator draws chunks by storage size, not number of lines):
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
# start and decoder stop. So most likely the 11MB fra-eng dataset is too small 
# to be representative for character-level language modelling (esp. at longer lengths).

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

class Sequence2Sequence(object):
    '''Sequence to sequence example in Keras (byte-level).

    Adapted from examples/lstm_seq2seq.py 
    (tutorial by Francois Chollet "A ten-minute introduction...") 
    with changes as follows:

    - use early stopping to prevent overfitting
    - use all data, sorted into increments by increasing window length
    - measure results not only on training set, but validation set as well
    - extended for use of full (large) dataset: training uses generator
      function (but not fit_generator because we want to have an automatic
      validation split for early stopping within in each increment)
    - preprocessing utilising Keras API: pad_sequences, to_categorical etc
      (but tokeniser not even necessary for byte strings)
    - add end-of-sequence symbol to encoder input in training and inference
    - based on byte level instead of character level (unlimited vocab)
    - padding needs to be taken into account by loss function: 
      mask padded samples using sample_weight zero
    - add preprocessing function for conveniently testing encoder
    - change first layer to bidirectional, add n more unidirectional LSTM layers
      (n configurable)
    - add beam search decoding (which enforces utf-8 via incremental decoder)
    - detect CPU vs GPU mode automatically
    - save/load weights separate from configuration (by recompiling model)
      in order to share weights between CPU and GPU model, 
      and between fixed and variable batchsize/length 

    Features still (very much) wanting of implementation:

    - attention
    - systematic hyperparameter treatment (deviations from Sutskever
      should be founded on empirical analysis): 
      HL width and depth, optimiser choice (RMSprop/SGD) and parameters,
      gradient clipping, decay and rate control, initialisers

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

    def __init__(self):
        self.batch_size = 64  # How many samples are trained together? (batch length)
        self.window_size = 6 # How many bytes are encoded at once? (sequence length)
        self.stateful = False # stateful encoder (implicit state transfer between batches) FIXME as soon as we have windows
        self.width = 320  # latent dimensionality of the encoding space (hidden layers length)
        self.depth = 4 # number of encoder/decoder layers stacked above each other (only 1st layer will be BLSTM)
        self.epochs = 100  # maximum number of epochs to train for (unless stopping early)
        self.beam_width = 4 # keep track of how many alternative sequences during decode_sequence_beam()?
        self.beam_width_out = self.beam_width # up to how many results can be drawn from generator decode_sequence_beam()?
        
        self.num_encoder_tokens = 256 # now utf8 bytes (including nul for non-output, tab for start, newline for stop)
        self.num_decoder_tokens = 256 # now utf8 bytes (including nul for non-output, tab for start, newline for stop)
        
        self.encoder_decoder_model = None
        self.encoder_model = None
        self.decoder_model = None
        
        self.status = 0 # empty / configured / trained?
    
    # for fit_generator()/predict_generator()/evaluate_generator() -- looping, but not shuffling
    def gen_data(self, filename, batch_size=None, reset_cb=None, get_size=False, get_edits=False, split=None, train=False):
        '''generate batches of vector data from text file
        
        Opens `filename` in binary mode, loops over it (unless `get_size`),
        producing one window of `batch_size` lines at a time
        (padding windows to a `self.window_size` multiple of the longest line, respectively).
        If stateful, calls `reset_cb` at the start of each line (if given) or resets model immediately (otherwise).
        Skips lines at `split` positions, depending on `train` (upper vs lower partition).
        Yields:
        - accumulated prediction error metrics if `get_edits`,
        - number of batches if `get_size`,
        - vector data batches (for fit_generator/evaluate_generator) otherwise.
        '''
        if not batch_size:
            batch_size = self.batch_size
        
        with open(filename, 'rb') as f:
            while True:
                source_texts = []
                target_texts = []
                batch_no = 0
                for line_no, line in enumerate(f):
                    if split is np.ndarray and (split[line_no] < 0.2) == train:
                        continue # data shared between training and validation: belongs to other generator
                    if len(source_texts) == 0 and not get_size: # start of batch
                        if self.stateful:
                            if reset_cb: # model controlled by callbacks (training)
                                reset_cb.reset("lines %d-%d" % (line_no, line_no+batch_size)) # inform callback
                            else: # model controlled by caller (batch prediction)
                                self.encoder_decoder_model.reset_states()
                                self.encoder_model.reset_states()
                                self.decoder_model.reset_states()
                    source_text, target_text = line.split(b'\t')
                    source_text = source_text + b'\n' # add end-of-sequence
                    target_text = target_text # start-of-sequence will be added window by window, end-of-sequence already preserved by file iterator
                    
                    # byte-align source and target text line, shelve them into successive fixed-size windows
                    source_seq = vocabulary.encodeSequence(Sequence(source_text.decode('utf-8')))
                    target_seq = vocabulary.encodeSequence(Sequence(target_text.decode('utf-8')))
                    score, alignments = aligner.align(source_seq, target_seq, backtrace=True)
                    alignment1 = vocabulary.decodeSequenceAlignment(alignments[0])
                    #print ('alignment score:', alignment1.score)
                    #print ('alignment rate:', alignment1.percentIdentity())
                    
                    # Produce batches of training data:
                    # - fixed `batch_size` lines (each line post-padded to maximum window multiple among batch)
                    # - fixed `self.window_size` samples (each window post-padded to fixed sequence length)
                    # with decoder windows twice as long as encoder windows.
                    # Yield window by window when full, then inform callback `reset_cb`.
                    # Callback will do `reset_states()` on all stateful layers of model at `on_batch_begin`.
                    # If stateful, also throw away partial batches at the very end of the file.
                    # Ensure that no window cuts through a Unicode codepoint, and the target window fits
                    # within twice the source window (always move partial characters to the next window).
                    try:
                        i = 0
                        j = 0
                        source_windows = [[]]
                        target_windows = [[b'\t'[0]]]
                        for source_char, target_char in alignment1:
                            source_bytes = source_char.encode('utf-8') if source_char != GAP_ELEMENT else b''
                            target_bytes = target_char.encode('utf-8') if target_char != GAP_ELEMENT else b''
                            source_len = len(source_bytes)
                            target_len = len(target_bytes)
                            if source_char != GAP_ELEMENT:
                                if i+source_len >= self.window_size:
                                    if len(target_windows[-1]) == 1:
                                        raise Exception("source window does not fit half the target window in alignment", alignment1, line.decode("utf-8"), source_windows, target_windows)
                                    i = 0
                                    j = 1
                                    source_windows.append([])
                                    target_windows.append([b'\t'[0]]) # add start-of-sequence (for this window)
                                source_windows[-1].extend(source_bytes)
                                i += source_len
                            if target_char != GAP_ELEMENT:
                                if j+target_len >= self.window_size*2:
                                    if len(source_windows[-1]) == 0:
                                        raise Exception("target window does not fit twice the source window in alignment", alignment1, line.decode("utf-8"), source_windows, target_windows)
                                    i = 0
                                    j = 1
                                    source_windows.append([])
                                    target_windows.append([b'\t'[0]]) # add start-of-sequence (for this window)
                                target_windows[-1].extend(target_bytes)
                                j += target_len
                        if j >= self.window_size*2:
                            raise Exception("target window does not fit twice the source window in alignment", alignment1, line.decode("utf-8"), source_windows, target_windows)
                    except Exception as e:
                        print(e)
                    
                    source_texts.append(source_windows)
                    target_texts.append(target_windows)
                    
                    if len(source_texts) == batch_size: # end of batch
                        max_windows = max([len(line) for line in source_texts])                
                        if get_size: # merely calculate number of batches that would be generated?
                            batch_no += max_windows
                            source_texts = []
                            target_texts = []
                            continue
                        
                        # yield windows...
                        for i in range(max_windows):
                            batch_no += 1
                            
                            # vectorize
                            encoder_input_sequences = list(map(bytearray,[line[i] if len(line)>i else b'' for line in source_texts]))
                            decoder_input_sequences = list(map(bytearray,[line[i] if len(line)>i else b'' for line in target_texts]))
                            # with windowing, we cannot use pad_sequences for zero-padding any more, 
                            # because zero bytes are a valid decoder input or output now (and different from zero input or output):
                            #encoder_input_sequences = pad_sequences(encoder_input_sequences, maxlen=self.window_size, padding='post')
                            #decoder_input_sequences = pad_sequences(decoder_input_sequences, maxlen=self.window_size*2, padding='post')
                            #encoder_input_data = np.eye(256, dtype=np.float32)[encoder_input_sequences,:]
                            #decoder_input_data = np.eye(256, dtype=np.float32)[decoder_input_sequences,:]
                            encoder_input_data = np.zeros((batch_size, self.window_size, self.num_encoder_tokens), dtype=np.float32)
                            decoder_input_data = np.zeros((batch_size, self.window_size*2, self.num_decoder_tokens), dtype=np.float32)
                            for j, (enc_seq, dec_seq) in enumerate(zip(encoder_input_sequences, decoder_input_sequences)):
                                if not len(enc_seq):
                                    continue # empty window (i.e. j-th line is shorter than others): true zero-padding (masked)
                                else:
                                    # encoder uses 256 dimensions (including zero byte):
                                    encoder_input_data[j*np.ones(len(enc_seq), dtype='int'), 
                                                       np.arange(len(enc_seq), dtype='int'), 
                                                       np.array(enc_seq, dtype='int')-1] = 1
                                    # zero bytes in encoder input become 1 at zero-byte dimension: indexed zero-padding (learned)
                                    padlen = self.window_size - len(enc_seq)
                                    decoder_input_data[j*np.ones(padlen, dtype='int'), 
                                                       np.arange(len(enc_seq), self.window_size, dtype='int'), 
                                                       np.zeros(padlen, dtype='int')] = 1
                                    # decoder uses 256 dimensions (including zero byte):
                                    decoder_input_data[j*np.ones(len(dec_seq), dtype='int'), 
                                                       np.arange(len(dec_seq), dtype='int'), 
                                                       np.array(dec_seq, dtype='int')] = 1
                                    # zero bytes in decoder input become 1 at zero-byte dimension: indexed zero-padding (learned)
                                    padlen = self.window_size*2 - len(dec_seq)
                                    decoder_input_data[j*np.ones(padlen, dtype='int'), 
                                                       np.arange(len(dec_seq), self.window_size*2, dtype='int'), 
                                                       np.zeros(padlen, dtype='int')] = 1
                            # teacher forcing:
                            decoder_output_data = np.roll(decoder_input_data,-1,axis=1) # output data will be ahead by 1 timestep
                            decoder_output_data[:,-1,:] = np.zeros(self.num_decoder_tokens) # delete+pad start token rolled in at the other end
                            
                            # index of padded samples, so we can mask them with the sample_weight parameter during fit() below
                            decoder_output_weights = np.ones(decoder_output_data.shape[:-1], dtype=np.float32)
                            decoder_output_weights[np.all(decoder_output_data == 0, axis=2)] = 0.

                            if get_edits: # calculate edits from decoding
                                assert batch_size == 1 # decoding works line-wise (with batches of alternatives)
                                assert j == 0 # corollary to that
                                if i == 0:
                                    self.encoder_model.reset_states()
                                    self.decoder_model.reset_states()
                                c_total, w_total = 0, 0
                                c_edits_ocr, w_edits_ocr = 0, 0
                                c_edits_greedy, w_edits_greedy = 0, 0
                                c_edits_beamed, w_edits_beamed = 0, 0
                                
                                source_seq = encoder_input_data[0]
                                target_seq = decoder_input_data[0]
                                # Take one sequence (part of the training/validation set) and try decoding it
                                source_text = source_texts[0][i].rstrip(bytes[0]).decode("utf-8", "strict")
                                target_text = target_texts[0][i].rstrip(bytes[0]).lstrip(b'\t').decode("utf-8", "strict")
                                decoded_text = self.decode_sequence_greedy(source_seq).rstrip(bytes[0]).decode("utf-8", "ignore")
                                try: # query only 1-best
                                    beamdecoded_text = next(self.decode_sequence_beam(source_seq)).rstrip(bytes[0]).decode("utf-8", "strict")
                                except StopIteration:
                                    print('no beam decoder result within processing limits for', source_text, target_text)
                                    continue # skip this line
                                print('Line', line_no, 'window', i)
                                print('Source input  from', 'training:' if train else 'test:    ', source_text)
                                print('Target output from', 'training:' if train else 'test:    ', target_text)
                                print('Target prediction (greedy): ', decoded_text)
                                print('Target prediction (beamed): ', beamdecoded_text)
                                
                                edits = edit_eval(source_text,target_text)
                                c_edits_ocr += edits
                                c_total += len(target_text)
                                edits = edit_eval(decoded_text,target_text)
                                c_edits_greedy += edits
                                edits = edit_eval(beamdecoded_text,target_text)
                                c_edits_beamed += edits
                                
                                decoded_tokens = decoded_text.split(" ")
                                beamdecoded_tokens = beamdecoded_text.split(" ")
                                source_tokens = source_text.split(" ")
                                target_tokens = target_text.split(" ")
                                
                                edits = edit_eval(source_tokens,target_tokens)
                                w_edits_ocr += edits
                                w_total += len(target_tokens)
                                edits = edit_eval(decoded_tokens,target_tokens)
                                w_edits_greedy += edits
                                edits = edit_eval(beamdecoded_tokens,target_tokens)
                                w_edits_beamed += edits
                                
                                yield (c_total, c_edits_ocr, c_edits_greedy, c_edits_beamed, w_total, w_edits_ocr, w_edits_greedy, w_edits_beamed)
                            else:
                                yield ([encoder_input_data, decoder_input_data], decoder_output_data)
                        
                        source_texts = []
                        target_texts = []
                if get_size: # return size, do not loop
                    yield batch_no
                elif get_edits: # do not loop
                    break
                else:
                    f.seek(0) # make sure the generator wraps around
    
    def configure(self):
        from keras.layers import Input, Dense, TimeDistributed
        from keras.layers import LSTM, CuDNNLSTM, Bidirectional, concatenate
        from keras.models import Model
        from keras import backend as K
        
        # automatically switch to CuDNNLSTM if CUDA GPU is available:
        has_cuda = K.backend() == 'tensorflow' and K.tensorflow_backend._get_available_gpus()
        print('using', 'GPU' if has_cuda else 'CPU', 'LSTM implementation to compile',
              'stateful' if self.stateful else 'stateless', 
              'model of depth', self.depth, 'width', self.width)
        lstm = CuDNNLSTM if has_cuda else LSTM
        
        ### Define training phase model
        
        # Set up an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        # Set up the encoder. We will discard encoder_outputs and only keep encoder_state_outputs.
        #dropout/recurrent_dropout does not seem to help (at least for small unidirectional encoder), go_backwards helps for unidirectional encoder with ~0.1 smaller loss on validation set (and not any slower) unless UTF-8 byte strings are used directly
        encoder_state_outputs = []
        for n in range(self.depth):
            args = {'return_state': True, 'return_sequences': (n < self.depth-1), 'stateful': self.stateful}
            if not has_cuda:
                args['recurrent_activation'] = 'sigmoid' # instead of default 'hard_sigmoid' which deviates from CuDNNLSTM
            layer = lstm(self.width, **args)
            if n == 0:
                encoder_outputs, fw_state_h, fw_state_c, bw_state_h, bw_state_c = Bidirectional(layer)(encoder_inputs)
                state_h = concatenate([fw_state_h, bw_state_h])
                state_c = concatenate([fw_state_c, bw_state_c])
            else:
                encoder_outputs, state_h, state_c = layer(encoder_outputs)
            encoder_state_outputs.extend([state_h, state_c])
        
        # Set up an input sequence and process it.
        decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        # Set up decoder to return full output sequences (so we can train in parallel),
        # to use encoder_state_outputs as initial state and return final states as well. 
        # We don't use those states in the training model, but we will use them 
        # for inference (see further below). 
        decoder_lstms = []
        for n in range(self.depth):
            args = {'return_state': True, 'return_sequences': True}
            if not has_cuda:
                args['recurrent_activation'] = 'sigmoid' # instead of default 'hard_sigmoid' which deviates from CuDNNLSTM
            layer = lstm(self.width*2 if n == 0 else self.width, **args)
            decoder_lstms.append(layer)
            decoder_outputs, _, _ = layer(decoder_inputs if n == 0 else decoder_outputs, 
                                          initial_state=encoder_state_outputs[2*n:2*n+2])
        decoder_dense = TimeDistributed(Dense(self.num_decoder_tokens, activation='softmax'))
        decoder_outputs = decoder_dense(decoder_outputs)

        # Bundle the model that will turn
        # `encoder_input_data` and `decoder_input_data` into `decoder_output_data`
        self.encoder_decoder_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        ## Define inference phase model
        # Here's the drill:
        # 1) encode source to retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        #    and a "start of sequence" as target token.
        # 3) repeat from 2, feeding back the target token 
        #    from output to input, and passing state
        
        # Re-use the training phase encoder unchanged 
        # (with result states as output).
        self.encoder_model = Model(encoder_inputs, encoder_state_outputs)
        
        # Set up decoder differently: with additional input
        # as initial state (not just encoder_state_outputs), and
        # keeping final states (instead of discarding), 
        # so we can pass states explicitly.
        decoder_state_inputs = []
        decoder_state_outputs = []
        for n in range(self.depth):
            state_h_in = Input(shape=(self.width*2 if n == 0 else self.width,))
            state_c_in = Input(shape=(self.width*2 if n == 0 else self.width,))
            decoder_state_inputs.extend([state_h_in, state_c_in])
            layer = decoder_lstms[n]
            decoder_outputs, state_h_out, state_c_out = layer(decoder_inputs if n == 0 else decoder_outputs, 
                                                              initial_state=decoder_state_inputs[2*n:2*n+2])
            decoder_state_outputs.extend([state_h_out, state_c_out])
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_state_inputs,
            [decoder_outputs] + decoder_state_outputs)
        
        ## Compile model
        self.encoder_decoder_model.compile(loss='categorical_crossentropy',
                                           optimizer='adam',
                                           sample_weight_mode='temporal') # sample_weight slows down slightly (20%)
        
        self.status = 1
    
    def train(self, filename):
        from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

        class ResetStatesCallback(Callback):
            '''Keras callback for stateful models to reset state between files.

            Callback to be called by `fit_generator()` or even `evaluate_generator()`:
            do `model.reset_states()` whenever generator sees EOF (on_batch_begin with self.eof),
            and between training and validation (on_batch_end with batch>=steps_per_epoch-1).
            '''
            def __init__(self):
                self.eof = False
                self.here = ''
                self.next = ''

            def reset(self, where):
                self.eof = True
                self.next = where

            def on_batch_begin(self, batch, logs={}):
                if self.eof:
                    # between training files
                    self.model.reset_states()
                    self.eof = False
                    self.here = self.next

            def on_batch_end(self, batch, logs={}):
                if logs.get('loss') > 25:
                    print('huge loss in', self.here, 'at', batch)
                if (self.params['do_validation'] and batch >= self.params['steps']-1):
                    # in fit_generator just before evaluate_generator
                    self.model.reset_states()
        
        # Run training
        callbacks = [EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='min'),
                     ModelCheckpoint('s2s_last.h5', monitor='val_loss', # to be able to replay long epochs (crash/overfitting)
                                     save_best_only=True, save_weights_only=True, mode='min')]
        if self.stateful: # reset states between batches of different lines/documents (controlled by generator)
            callbacks.append(ResetStatesCallback())
        
        # todo: shuffle lines
        # count how many batches the generator would return per epoch
        with open(filename, 'rb') as f:
            num_lines = sum(1 for line in f)
        split_rand = np.random.uniform(0, 1, (num_lines,)) # reserve split fraction at random line numbers
        training_epoch_size = next(self.gen_data(filename, train=True, split=split_rand, get_size=True))
        validation_epoch_size = next(self.gen_data(filename, train=False, split=split_rand, get_size=True))
        self.encoder_decoder_model.fit_generator(self.gen_data(filename, train=True, split=split_rand, reset_cb=callbacks[-1]),
                                                 steps_per_epoch=training_epoch_size, epochs=self.epochs,
                                                 validation_data=self.gen_data(filename, train=False, split=split_rand),
                                                 validation_steps=validation_epoch_size, verbose=1, callbacks=callbacks)
        
        self.state = 2
    
    def load_config(self, filename):
        config = pickle.load(open(filename, mode='rb'))
        self.width = config['width']
        self.depth = config['depth']
        self.stateful = config['stateful']
    
    def save_config(self, filename):
        assert self.status > 0 # already compiled
        config = {'width': self.width, 'depth': self.depth, 'stateful': self.stateful}
        pickle.dump(config, open(filename, mode='wb'))
    
    def load_weights(self, filename):
        assert self.status > 0 # already compiled
        self.encoder_decoder_model.load_weights(filename)
        self.status = 2
    
    def save_weights(self, filename):
        assert self.status > 1 # already trained
        self.encoder_decoder_model.save_weights(filename)
    
    def decode_sequence_greedy(self, source_seq):
        '''to be called on 1 line of input for each (non-zero) window
        '''
        # reset must be done at line break (by caller)
        #self.encoder_model.reset_states()
        #self.decoder_model.reset_states()
       
        # Encode the source as state vectors.
        states_value = self.encoder_model.predict(np.expand_dims(source_seq, axis=0))
        
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        #target_seq = np.zeros((1, 1))
        # Populate the first character of target sequence with the start character.
        #target_seq[0, 0, target_token_index['\t']-1] = 1.
        target_seq[0, 0, b'\t'[0]] = 1.
        #target_seq[0, 0] = b'\t'[0]
        
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_text = b''
        while not stop_condition:
            output = self.decoder_model.predict_on_batch([target_seq] + states_value)
            output_scores = output[0]
            output_states = output[1:]
            
            # Sample a token
            sampled_token_index = np.argmax(output_scores[0, -1, :])
            #sampled_char = reverse_target_char_index[sampled_token_index+1]
            sampled_char = bytes([sampled_token_index])
            decoded_text += sampled_char
            
            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == b'\n' or
                len(decoded_text) >= self.window_size*2):
                stop_condition = True
            
            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            #target_seq = np.zeros((1, 1))
            target_seq[0, 0, sampled_token_index] = 1.
            #target_seq[0, 0] = sampled_token_index+1
            # feeding back the softmax vector directly vastly deteriorates predictions (because only identity vectors are presented during training)
            # but we should add beam search (keep n best overall paths, add k best predictions here)
            
            # Update states
            states_value = list(output_states)
        
        return decoded_text.strip(bytes[0]) # strip trailing but keep intermediate zero bytes
    
    def decode_sequence_beam(self, source_seq):
        '''to be called on 1 line of input for each (non-zero) window
        '''
        from bisect import insort_left
        
        decoder = codecs.getincrementaldecoder('utf8')()
        decoder.decode(b'\t')
        
        #self.encoder_model.reset_states()
        #self.decoder_model.reset_states()
        next_fringe = [Node(parent=None,
                            state=self.encoder_model.predict(np.expand_dims(source_seq, axis=0)), # layer list of state arrays
                            value=b'\t'[0], # start symbol byte
                            cost=0.0,
                            extras=decoder.getstate())]
        hypotheses = []
        
        # generator will raise StopIteration if hypotheses is still empty after loop
        MAX_BATCHES = self.window_size*4 # how many batches (i.e. byte-hypotheses) will be processed per window?
        for l in range(MAX_BATCHES):
            # try:
            #     next(n for n in next_fringe if all(np.array_equal(x,y) for x,y in zip(nonbeam_states[:l+1], [s.state for s in n.to_sequence()])))
            # except StopIteration:
            #     print('greedy result falls off beam search at pos', l, nonbeam_seq[:l+1])
            fringe = []
            while next_fringe:
                n = next_fringe.pop()
                if n.value == b'\n'[0] or n.length == self.window_size*2: # end-of-sequence symbol or window full?
                    hypotheses.append(n)
                    #print('found new solution', bytes([i for i in n.to_sequence_of_values()]).decode("utf-8", "ignore"))
                else: # normal step
                    fringe.append(n)
                    #print('added new hypothesis', bytes([i for i in n.to_sequence_of_values()]).decode("utf-8", "ignore"))
                if len(fringe) >= self.batch_size:
                    break # enough for one batch
            if len(hypotheses) >= self.beam_width_out:
                break # done
            if not fringe:
                break # will give StopIteration unless we have some results already
            
            # use fringe leaves as minibatch, but with only 1 timestep
            target_seq = np.expand_dims(np.eye(256, dtype=np.float32)[[n.value for n in fringe], :], axis=1) # add time dimension
            states_val = [np.vstack([n.state[layer] for n in fringe]) for layer in range(len(fringe[0].state))] # stack layers across batch
            output = self.decoder_model.predict_on_batch([target_seq] + states_val)
            scores_output = output[0][:,-1] # only last timestep
            scores_output_order = np.argsort(scores_output, axis=1) # still in reverse order (worst first)
            #[:,-self.beam_width:] # for fixed beam width
            states_output = list(output[1:]) # from (layers) tuple
            for i, n in enumerate(fringe): # iterate over batch (1st dim)
                scores = scores_output[i,:]
                scores_order = scores_output_order[i,:]
                highest = scores[scores_order[-1]]
                beampos = 256 - np.searchsorted(scores[scores_order], 0.05 * highest) # variable beam width
                states = [layer[i:i+1] for layer in states_output] # unstack layers for current sample
                logscores = -np.log(scores[scores_order])
                pos = 0
                for best, logscore in zip(reversed(scores_order), reversed(logscores)): # follow up on beam_width best predictions
                    # todo: disallow zero byte not followed by zero byte
                    decoder.setstate(n.extras)
                    try:
                        decoder.decode(bytes([best]))
                        n_new = Node(parent=n, state=states, value=best, cost=logscore, extras=decoder.getstate())
                        insort_left(next_fringe, n_new)
                        pos += 1
                    except UnicodeDecodeError:
                        pass # ignore this alternative
                    if pos > beampos: # less than one tenth the highest probability?
                        break # ignore further alternatives
            if len(next_fringe) > MAX_BATCHES * self.batch_size: # more than can ever be processed within limits?
                next_fringe = next_fringe[-MAX_BATCHES*self.batch_size:] # to save memory, keep only best
        
        hypotheses.sort(key=lambda n: n.cum_cost)
        for hypothesis in hypotheses[:self.beam_width_out]:
            indices = hypothesis.to_sequence_of_values()
            byteseq = bytes([i for i in indices]).strip(bytes[0]) # strip trailing but keep intermediate zero bytes
            yield byteseq
    


class Node(object):
    def __init__(self, parent, state, value, cost, extras):
        super(Node, self).__init__()
        self.value = value # byte
        self.parent = parent # parent Node, None for root
        self.state = state # recurrent layer hidden state
        self.cum_cost = parent.cum_cost + cost if parent else cost # e.g. -log(p) of sequence up to current node (including)
        self.length = 1 if parent is None else parent.length + 1
        self.extras = extras
        self._norm_cost = self.cum_cost / self.length
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
    
    # for sort order, use cumulative costs relative to length
    # (in order to get a fair comparison across different lengths,
    #  and hence, depth-first search), and use inverse order
    # (so the faster pop() can be used)
    def __lt__(self, other):
        return other._norm_cost < self._norm_cost
    def __le__(self, other):
        return other._norm_cost <= self._norm_cost
    def __eq__(self, other):
        return other._norm_cost == self._norm_cost
    def __ne__(self, other):
        return other._norm_cost != self._norm_cost
    def __gt__(self, other):
        return other._norm_cost > self._norm_cost
    def __ge__(self, other):
        return other._norm_cost >= self._norm_cost
    

s2s = Sequence2Sequence()
vocabulary = Vocabulary() # merely for byte alignment of input lines (for windowing)
scoring = SimpleScoring(2,-1) # match score, mismatch score
aligner = StrictGlobalSequenceAligner(scoring,-2) # gap score

data_path = '../../daten/dta19-reduced/traindata.Fraktur4-gt.filtered.augmented.txt' # training and validation set csv
data_path2 = '../../daten/dta19-reduced/testdata.Fraktur4-gt.filtered.txt' # test set csv

model_filename = 's2s.Fraktur4.d%d.w%04d.h5' % (s2s.depth, s2s.width)
if len(sys.argv) > 1:
    model_filename = sys.argv[1]
config_filename = model_filename.rstrip('.h5') + '.pkl'
if len(sys.argv) > 2:
    s2s.beam_width = int(sys.argv[2])
    s2s.beam_width_out = s2s.beam_width

if isfile(config_filename) and isfile(model_filename):
    print('Loading model', model_filename, config_filename)
    #model = s2s.encoder_decoder_model.load_model(model_filename)
    s2s.load_config(config_filename)
    s2s.configure()
    s2s.load_weights(model_filename)
else:
    s2s.configure()
    s2s.train(data_path)
    
    # Save model
    print('Saving model', model_filename, config_filename)
    s2s.save_config(config_filename)
    if isfile('s2s_last.h5'):
        rename('s2s_last.h5', model_filename)
    else:
        s2s.save_weights(model_filename)


#with open(other_path, 'rb') as f:
#    loss = s2s.encoder_decoder_model.evaluate_generator(gen_data(f), workers=4, verbose=1)
#    print('Test loss:', loss)
#would be greedy:
#output = s2s.encoder_decoder_model.predict_on_generator(gen_data(data_path2))
c_total = 0
c_edits_ocr = 0
c_edits_greedy = 0
c_edits_beamed = 0
w_total = 0
w_edits_ocr = 0
w_edits_greedy = 0
w_edits_beamed = 0
epoch_size = next(s2s.gen_data(data_path2, batch_size=1, get_size=True))
with click.progressbar(length=epoch_size) as bar:
    for (c_total_batch, c_edits_ocr_batch, c_edits_greedy_batch, c_edits_beamed_batch,
         w_total_batch, w_edits_ocr_batch, w_edits_greedy_batch, w_edits_beamed_batch) \
        in s2s.gen_data(data_path2, batch_size=1, get_edits=True): # get windows for only 1 line at a time
        bar.update(1)
        c_total += c_total_batch
        c_edits_ocr += c_edits_ocr_batch
        c_edits_greedy += c_edits_greedy_batch
        c_edits_beamed += c_edits_beamed_batch
        w_total += w_total_batch
        w_edits_ocr += w_edits_ocr_batch
        w_edits_greedy += w_edits_greedy_batch
        w_edits_beamed += w_edits_beamed_batch

print("CER OCR: {}".format(c_edits_ocr / c_total))
print("CER greedy: {}".format(c_edits_greedy / c_total))
print("CER beamed: {}".format(c_edits_beamed / c_total))
print("WER OCR: {}".format(w_edits_ocr / w_total))
print("WER greedy: {}".format(w_edits_greedy / w_total))
print("WER beamed: {}".format(w_edits_beamed / w_total))

def encode_text(source_text):
    # source_sequence = np.zeros((1, max_encoder_seq_length, s2s.num_encoder_tokens), dtype='float32')
    # for t, char in enumerate(source_text):
    #     source_sequence[0, t, source_token_index[char]] = 1
    # return source_sequence
    #return to_categorical(pad_sequences(source_tokenizer.texts_to_sequences([source_text]), maxlen=max_encoder_seq_length, padding='pre'), num_classes=s2s.num_encoder_tokens+1)[:,:,1:] # remove separate dimension for zero/padding
    #return to_categorical(pad_sequences([list(map(bytearray,source_text))], maxlen=max_encoder_seq_length, padding='pre'), num_classes=s2s.num_encoder_tokens+1)[:,:,1:] # remove separate dimension for zero/padding
    #return to_categorical(list(map(bytearray,[source_text + b'\n'])), num_classes=s2s.num_encoder_tokens+1)[:,:,1:]
    #return bytearray(source_text+b'\n')
    return np.eye(256, dtype=np.float32)[bytearray(source_text+b'\n'),:]

print("usage example:\n# for reading in s2s.decode_sequence_beam(encode_text(b'hello world!')):\n#     print(reading.decode('utf-8'))")
