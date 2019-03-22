#!/usr/bin/python -i
# -*- coding: utf-8
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
import math
import signal
import logging
import numpy as np
import h5py
# these should all be wrapped in functions:
import pickle
import click
import threading
from keras.callbacks import Callback

# load pythonrc even with -i
if 'PYTHONSTARTUP' in environ:
    exec(open(environ['PYTHONSTARTUP']).read())
if not 'TF_CPP_MIN_LOG_LEVEL' in environ:
    environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

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

class Alignment(object):
    def __init__(self, GAP_ELEMENT):
        self.GAP_ELEMENT = GAP_ELEMENT
        # alignment for windowing...
        ## python-alignment is impractical with long or heavily deviating sequences (see github issues 9, 10, 11):
        #import alignment.sequence
        #alignment.sequence.GAP_ELEMENT = self.GAP_ELEMENT # override default
        #from alignment.sequence import Sequence, GAP_ELEMENT
        #from alignment.vocabulary import Vocabulary
        #from alignment.sequencealigner import SimpleScoring, StrictGlobalSequenceAligner
        # Levenshtein scoring:
        #self.scoring = SimpleScoring(2,-1) # match score, mismatch score
        #self.aligner = StrictGlobalSequenceAligner(scoring,-2) # gap score
        # Levenshtein-like scoring with 0.1 distance within typical OCR confusion classes (to improve alignment quality; to reduce error introduced by windowing):
        # class OCRScoring(SimpleScoring):
        #     def __init__(self):
        #         super(OCRScoring, self).__init__(0,-1) # match score, mismatch score (Levenshtein-like)
        #         self.classes = [[u"a", u"ä", u"á", u"â", u"à", u"ã"],
        #                         [u"o", u"ö", u"ó", u"ô", u"ò", u"õ"],
        #                         [u"u", u"ü", u"ú", u"û", u"ù", u"ũ"],
        #                         [u"A", u"Ä", u"Á", u"Â", u"À", u"Ã"],
        #                         [u"O", u"Ö", u"Ó", u"Ô", u"Ò", u"Õ"],
        #                         [u"U", u"Ü", u"Ú", u"Û", u"Ù", u"Ũ"],
        #                         [0, u"ͤ"],
        #                         [u'"', u"“", u"‘", u"'", u"’", u"”"],
        #                         [u',', u'‚', u'„'],
        #                         [u'-', u'‐', u'—', u'–', u'_'],
        #                         [u'=', u'⸗', u'⹀'],
        #                         [u'ſ', u'f', u'ß'], #s?
        #                         [u"c", u"<", u"e"]]
        #         self.table = {}
        #         for c in self.classes:
        #             for i in c:
        #                 for j in c:
        #                     if i==j:
        #                         self.table[(i,j)] = 0.0
        #                     else:
        #                         self.table[(i,j)] = 0.1
        #     def __call__(self, firstElement, secondElement):
        #         if (firstElement,secondElement) in self.table:
        #             return self.table[(firstElement,secondElement)]
        #         else:
        #             return super(OCRScoring, self).__call__(firstElement, secondElement)
        # 
        # self.scoring = OCRScoring()
        # self.aligner = StrictGlobalSequenceAligner(scoring,-1) # gap score
        
        ## edlib does not work on Unicode (non-ASCII strings)
        # import edlib

        ## difflib is optimised for visual comparisons (Ratcliff-Obershelp), not minimal distance (Levenshtein):
        from difflib import SequenceMatcher
        self.matcher = SequenceMatcher(isjunk=None, autojunk=False)
        
        ## edit_distance is impractical with long sequences, even if very similar (GT lines > 1000 characters, see github issue 6)
        # from edit_distance.code import SequenceMatcher # similar API to difflib.SequenceMatcher
        # def char_similar(a, b):
        #     return (a == b or (a,b) in table)
        # self.matcher = SequenceMatcher(test=char_similar)
        
        self.source_text = []
        self.target_text = []
    
    def set_seqs(self, source_text, target_text):
        ## code for python_alignment:
        #vocabulary = Vocabulary() # inefficient, but helps keep search space smaller if independent for each line
        #self.source_seq = vocabulary.encodeSequence(Sequence(source_text))
        #self.target_seq = vocabulary.encodeSequence(Sequence(target_text))
        
        ## code for edlib:
        #self.edres = edlib.align(source_text, target_text, mode='NW', task='path',
        #                         k=max(len(source_text),len(target_text))*2)
        
        ## code for difflib/edit_distance:
        self.matcher.set_seqs(source_text, target_text)
        
        self.source_text = source_text
        self.target_text = target_text
        
    
    def is_bad(self):
        ## code for python_alignment:
        #score = self.aligner.align(self.source_seq, self.target_seq)
        #if score < -10 and score < 5-len(source_text):
        #    return True
        
        ## code for edlib:
        # assert self.edres
        # if self.edres['editDistance'] < 0:
        #    return True

        ## code for difflib/edit_distance:
        # self.matcher = difflib_matcher if len(source_text) > 4000 or len(target_text) > 4000 else editdistance_matcher
        
        # if self.matcher.distance() > 10 and self.matcher.distance() > len(self.source_text)-5:
        if self.matcher.quick_ratio() < 0.5 and len(self.source_text) > 5:
            return True
        else:
            return False
    
    def get_best_alignment(self):
        ## code for identity alignment (for GT-only training; faster, no memory overhead)
        # alignment1 = zip(source_text, target_text)
        
        ## code for python_alignment:
        #score, alignments = self.aligner.align(self.source_seq, self.target_seq, backtrace=True)
        #alignment1 = vocabulary.decodeSequenceAlignment(alignments[0])
        #alignment1 = zip(alignment1.first, alignment1.second)
        #print ('alignment score:', alignment1.score)
        #print ('alignment rate:', alignment1.percentIdentity())
        
        ## code for edlib:
        # assert self.edres
        # alignment1 = []
        # n = ""
        # source_k = 0
        # target_k = 0
        # for c in self.edres['cigar']:
        #     if c.isdigit():
        #         n = n + c
        #     else:
        #         i = int(n)
        #         n = ""
        #         if c in "=X": # identity/substitution
        #             alignment1.extend(zip(self.source_text[source_k:source_k+i], self.target_text[target_k:target_k+i]))
        #             source_k += i
        #             target_k += i
        #         elif c == "I": # insert into target
        #             alignment1.extend(zip(self.source_text[source_k:source_k+i], [self.GAP_ELEMENT]*i))
        #             source_k += i
        #         elif c == "D": # delete from target
        #             alignment1.extend(zip([self.GAP_ELEMENT]*i, self.target_text[target_k:target_k+i]))
        #             target_k += i
        #         else:
        #             raise Exception("edlib returned invalid CIGAR opcode", c)
        # assert source_k == len(self.source_text)
        # assert target_k == len(self.target_text)
        
        ## code for difflib/edit_distance:
        alignment1 = []
        for op, source_begin, source_end, target_begin, target_end in self.matcher.get_opcodes():
            if op == 'equal':
                alignment1.extend(zip(self.source_text[source_begin:source_end],
                                      self.target_text[target_begin:target_end]))
            elif op == 'replace': # not really substitution:
                delta = source_end-source_begin-target_end+target_begin
                #alignment1.extend(zip(self.source_text[source_begin:source_end] + [self.GAP_ELEMENT]*(-delta),
                #                      self.target_text[target_begin:target_end] + [self.GAP_ELEMENT]*(delta)))
                if delta > 0: # replace+delete
                    alignment1.extend(zip(self.source_text[source_begin:source_end-delta],
                                          self.target_text[target_begin:target_end]))
                    alignment1.extend(zip(self.source_text[source_end-delta:source_end],
                                          [self.GAP_ELEMENT]*(delta)))
                if delta <= 0: # replace+insert
                    alignment1.extend(zip(self.source_text[source_begin:source_end],
                                          self.target_text[target_begin:target_end+delta]))
                    alignment1.extend(zip([self.GAP_ELEMENT]*(-delta),
                                          self.target_text[target_end+delta:target_end]))
            elif op == 'insert':
                alignment1.extend(zip([self.GAP_ELEMENT]*(target_end-target_begin),
                                      self.target_text[target_begin:target_end]))
            elif op == 'delete':
                alignment1.extend(zip(self.source_text[source_begin:source_end],
                                      [self.GAP_ELEMENT]*(source_end-source_begin)))
            else:
                raise Exception("difflib returned invalid opcode", op, "in", self.source_text, self.target_text)
        assert source_end == len(self.source_text)
        assert target_end == len(self.target_text)

        return alignment1
    
    def get_levenshtein_distance(self, source_text, target_text):
        # alignment for evaluation only...
        import editdistance
        d = editdistance.eval(source_text, target_text)
        l = len(target_text)
        return d, l
    
    def get_adjusted_distance(self, source_text, target_text):
        self.set_seqs(source_text, target_text)
        alignment = self.get_best_alignment()
        d = 0 # distance
        
        umlauts = {u"ä": "a", u"ö": "o", u"ü": "u"} # for example
        #umlauts = {}
        
        source_umlaut = ''
        target_umlaut = ''
        for source_sym, target_sym in alignment:
            #print(source_sym, target_sym)
            
            if source_sym == target_sym:
                if source_umlaut: # previous source is umlaut non-error
                    source_umlaut = False # reset
                    d += 1.0 # one full error (mismatch)
                elif target_umlaut: # previous target is umlaut non-error
                    target_umlaut = False # reset
                    d += 1.0 # one full error (mismatch)
            else:
                if source_umlaut: # previous source is umlaut non-error
                    source_umlaut = False # reset
                    if (source_sym == self.GAP_ELEMENT and
                        target_sym == u"\u0364"): # diacritical combining e
                        d += 1.0 # umlaut error (umlaut match)
                        #print('source umlaut match', a)
                    else:
                        d += 2.0 # two full errors (mismatch)
                elif target_umlaut: # previous target is umlaut non-error
                    target_umlaut = False # reset
                    if (target_sym == self.GAP_ELEMENT and
                        source_sym == u"\u0364"): # diacritical combining e
                        d += 1.0 # umlaut error (umlaut match)
                        #print('target umlaut match', a)
                    else:
                        d += 2.0 # two full errors (mismatch)
                elif source_sym in umlauts and umlauts[source_sym] == target_sym:
                    source_umlaut = True # umlaut non-error
                elif target_sym in umlauts and umlauts[target_sym] == source_sym:
                    target_umlaut = True # umlaut non-error
                else:
                    d += 1.0 # one full error (non-umlaut mismatch)
        if source_umlaut or target_umlaut: # previous umlaut error
            d += 1.0 # one full error
        
        #length_reduction = max(source_text.count(u"\u0364"), target_text.count(u"\u0364"))
        return d, len(target_text) # d, len(alignment) - length_reduction # distance and adjusted length

class Sequence2Sequence(object):
    '''Sequence to sequence example in Keras (byte-level).

    Adapted from examples/lstm_seq2seq.py 
    (tutorial by Francois Chollet "A ten-minute introduction...") 
    with changes as follows:

    - use early stopping to prevent overfitting
    - use all data, sorted into increments by increasing window length
    - measure results not only on training set, but validation set as well
    - extended for use of full (large) dataset: training uses fit_generator
      with same generator function called twice split validation data for 
      early stopping (to prevent overfitting)
    - based on byte level instead of character level (unlimited vocab)
    - efficient preprocessing
    - add end-of-sequence symbol to encoder input in training and inference
    - padding needs to be taken into account by loss function: 
      mask padded samples using sample_weight zero
    - add runtime preprocessing function for conveniently testing encoder
    - change first layer to bidirectional, add more unidirectional LSTM layers
      (depth width and length configurable)
    - add beam search decoding (which enforces utf-8 via incremental decoder)
    - detect CPU vs GPU mode automatically
    - save/load weights separate from configuration (by recompiling model)
      in order to share weights between CPU and GPU model, 
      and between fixed and variable batchsize/length 
    - evaluate word and character error rates on separate dataset
    - window-based processing (with byte alignment of training data)
    - stateful encoder mode

    Features still (very much) wanting of implementation:

    - attention or at least peeking (not just final-initial transfer)
    - stateful decoder mode (in non-transfer part of state function)
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

    # References

    - Sequence to Sequence Learning with Neural Networks
        https://arxiv.org/abs/1409.3215
    - Learning Phrase Representations using
        RNN Encoder-Decoder for Statistical Machine Translation
        https://arxiv.org/abs/1406.1078
    '''

    def __init__(self):
        # model parameters
        self.batch_size = 64  # How many samples are trained together? (batch length)
        self.window_length = 7 # How many bytes are encoded at once? (sequence length)
        self.stateful = False # stateful encoder (implicit state transfer between batches)
        self.width = 320  # latent dimensionality of the encoding space (hidden layer state)
        self.depth = 2 # number of encoder/decoder layers stacked above each other (only 1st layer will be BLSTM)
        self.num_encoder_tokens = 256 # now utf8 bytes (including nul for non-output, tab for start, newline for stop)
        self.num_decoder_tokens = 256 # now utf8 bytes (including nul for non-output, tab for start, newline for stop)
        self.residual_connections = False # add input to output in encoder and decoder HL
        self.deep_bidirectional_encoder = False # encoder HL are all BLSTM cross-summarizing forward and backward outputs (as -encoder_type bdrnn in Open-NMT)
        self.bridge_dense = False # use a FFNN to map encoder final states to decoder initial states instead of copy
        
        # training parameters
        self.epochs = 100  # maximum number of epochs to train for (unless stopping early by validation loss)
        self.lm_loss = False # train with additional output (unweighted sum loss) from LM, defined with tied decoder weights and same input but not conditioned on encoder output (applies to encoder_decoder_model only, does not affect encoder_model and decoder_model)
        self.scheduled_sampling = None # 'linear'/'sigmoid'/'exponential'/None # train with additional output from self-loop (softmax feedback) instead of teacher forcing (with loss weighted by given curve across epochs), defined with tied weights and same encoder output (applies to encoder_decoder_model only, does not affect encoder_model and decoder_model)
        self.dropout = 0.2 # rate of dropped input connections in encoder and decoder HL during training
        
        # inference parameters
        self.beam_width = 4 # keep track of how many alternative sequences during decode_sequence_beam()?
        self.beam_width_out = self.beam_width # up to how many results can be drawn from generator decode_sequence_beam()?

        self.graph = None # for tf access from multiple threads
        self.encoder_decoder_model = None # combined model for training
        self.encoder_model = None # separate model for inference
        self.decoder_model = None # separate model for inference
        self.aligner = Alignment(0) # aligner (for windowing and/or evaluation) with internal state
        
        self.status = 0 # empty / configured / trained?
    
    # for fit_generator()/predict_generator()/evaluate_generator()/standalone -- looping, but not shuffling
    def gen_data(self, filename, split=None, train=False, reset_cb=None):
        '''generate batches of vector data from text file
        
        Open `filename` in binary mode, loop over it producing one window
        of batch_size lines at a time.
        Pad windows to a `self.window_length` multiple of the longest line,
        respectively.
        If stateful, call `reset_cb` at the start of each line (if given)
        or resets model directly (otherwise).
        Skip lines at `split` positions (if given), depending on `train`
        (upper vs lower partition).
        Yields vector data batches (for fit_generator/evaluate_generator).
        '''
        epoch = 0
        lock = threading.Lock()
        for batch in self.gen_windows(filename, True, split, train):
            if not batch:
                epoch += 1
                yield False # signal end of epoch to autosized fit/evaluate
                if train and self.scheduled_sampling:
                    # prepare empirical scheduled sampling (i.e. without proper gradient)
                    a = 3 # attenuation: 10 enters saturation at about 10 percent of self.epochs
                    if self.scheduled_sampling == 'linear':
                        sample_ratio = a*(epoch-1)/(self.epochs-1)
                    elif self.scheduled_sampling == 'sigmoid':
                        sample_ratio = 1/(1+math.exp(5-10*a*epoch/self.epochs))
                    elif self.scheduled_sampling == 'exponential':
                        sample_ratio = 1-0.9**(50*a*epoch/self.epochs)
                    else:
                        raise Exception('unknown function "%s" for scheduled sampling' % self.scheduled_sampling)
                    #print('sample ratio for this epoch:', sample_ratio)
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
                    encoder_input_data, decoder_input_data, decoder_output_data, decoder_output_weights = (
                        self.vectorize_windows(
                            [line[i] if len(line)>i else b'' for line in source_lines],
                            [line[i] if len(line)>i else b'' for line in target_lines],
                            [line[i] if len(line)>i else [] for line in sourceconf_lines] \
                            if sourceconf_lines else None))
                    # yield source/target data to keras consumer loop (fit/evaluate)
                    if line_schedules: # and epoch > 1:
                        # calculate greedy/beamed decoder output to yield as as decoder input
                        window_nonempty = np.array(list(map(lambda target: len(target)>i, target_lines))) # avoid lines with empty windows
                        data_nonempty = np.logical_not(np.any(np.all(decoder_input_data == 0, axis=2), axis=1))
                        assert np.array_equal(data_nonempty, window_nonempty), (
                            "unexpected zero window %d: %s" % 
                            (i, target_lines[np.nonzero(np.not_equal(window_nonempty, data_nonempty))[0][0]]))
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
                                #print('sampled %02d lines for window %d' % (np.count_nonzero(indexes), i))
                    if self.status > 1: # was already pre-trained?
                        # sample_weight quickly causes getting stuck with NaN,
                        # both in gradient updates and weights (regardless of
                        # loss function, optimizer, gradient clipping, CPU or GPU)
                        # when re-training, so disable
                        yield ([encoder_input_data, decoder_input_data], decoder_output_data)
                    else:
                        yield ([encoder_input_data, decoder_input_data], decoder_output_data,
                               decoder_output_weights)
                lock.release()
                    
    def vectorize_windows(self, encoder_input_sequences, decoder_input_sequences, encoder_conf_sequences=None):
        encoder_input_sequences = list(map(bytearray, encoder_input_sequences))
        decoder_input_sequences = list(map(bytearray, decoder_input_sequences))
        # with windowing, we cannot use pad_sequences for zero-padding any more,
        # because zero bytes are a valid decoder input or output now (and different from zero input or output):
        encoder_input_data  = np.zeros((self.batch_size, self.window_length, self.num_encoder_tokens),
                                       dtype=np.float32 if encoder_conf_sequences else np.int8)
        decoder_input_data  = np.zeros((self.batch_size, self.window_length*1+1, self.num_decoder_tokens), dtype=np.int8)
        decoder_output_data = np.zeros((self.batch_size, self.window_length*1+1, self.num_decoder_tokens), dtype=np.int8)
        for j, (enc_seq, dec_seq) in enumerate(zip(encoder_input_sequences, decoder_input_sequences)):
            if len(enc_seq):
                # encoder uses 256 dimensions (including zero byte):
                encoder_input_data[j*np.ones(len(enc_seq), dtype=np.int8), 
                                   np.arange(len(enc_seq), dtype=np.int8), 
                                   np.array(enc_seq, dtype=np.int8)] = 1
                if encoder_conf_sequences: # binary input with OCR confidence?
                    encoder_input_data[j*np.ones(len(enc_seq), dtype=np.int8), 
                                       np.arange(len(enc_seq), dtype=np.int8), 
                                       np.array(enc_seq, dtype=np.int8)] = np.array(
                                           encoder_conf_sequences[j], dtype=np.float32)
                # zero bytes in encoder input become 1 at zero-byte dimension: indexed zero-padding (learned)
                padlen = self.window_length - len(enc_seq)
                encoder_input_data[j*np.ones(padlen, dtype=np.int8),
                                   np.arange(len(enc_seq), self.window_length, dtype=np.int8),
                                   np.zeros(padlen, dtype=np.int8)] = 1
            else:
                pass # empty window (i.e. j-th line is shorter than others): true zero-padding (masked)
            if len(dec_seq):
                # decoder uses 256 dimensions (including zero byte):
                decoder_input_data[j*np.ones(len(dec_seq), dtype=np.int8),
                                   np.arange(len(dec_seq), dtype=np.int8),
                                   np.array(dec_seq, dtype=np.int8)] = 1
                # zero bytes in decoder input become 1 at zero-byte dimension: indexed zero-padding (learned)
                padlen = self.window_length*1+1 - len(dec_seq)
                decoder_input_data[j*np.ones(padlen, dtype=np.int8),
                                   np.arange(len(dec_seq), self.window_length*1+1, dtype=np.int8),
                                   np.zeros(padlen, dtype=np.int8)] = 1
                # teacher forcing:
                decoder_output_data[j*np.ones(len(dec_seq)-1, dtype=np.int8),
                                    np.arange(len(dec_seq)-1, dtype=np.int8),
                                    np.array(dec_seq[1:], dtype=np.int8)] = 1
                decoder_output_data[j, len(dec_seq)-1, 0] = 1
                decoder_output_data[j*np.ones(padlen, dtype=np.int8),
                                    np.arange(len(dec_seq), self.window_length*1+1, dtype=np.int8),
                                    np.zeros(padlen, dtype=np.int8)] = 1
            else:
                pass # empty window (i.e. j-th line is shorter than others): true zero-padding (masked)
            
        # index of padded samples, so we can mask them
        # with the sample_weight parameter during fit() below
        decoder_output_weights = np.ones(decoder_output_data.shape[:-1], dtype=np.float32)
        decoder_output_weights[np.all(decoder_output_data == 0, axis=2)] = 0. # true zero (empty window)
        #decoder_output_weights[decoder_output_data[:,:,0] == 1] = 0. # padded zero (partial window)
        if self.lm_loss:
            lm_output_weights = np.ones(decoder_output_data.shape[:-1], dtype=np.float32)
            lm_output_weights[np.all(decoder_output_data == 0, axis=2)] = 0. # true zero
            lm_output_weights[decoder_output_data[:,:,0] == 1] = 0. # padded zero
            decoder_output_data = [decoder_output_data, decoder_output_data] # 2 outputs, 1 combined loss
            decoder_output_weights = [decoder_output_weights, lm_output_weights]
        
        return encoder_input_data, decoder_input_data, decoder_output_data, decoder_output_weights
    
    def gen_windows(self, filename, repeat=True, split=None, train=False):
        split_ratio = 0.2
        with_confidence = filename.endswith('.pkl')
        with open(filename, 'rb') as fd:
            epoch = 0
            if with_confidence:
                fd = pickle.load(fd) # read once
            while True:
                source_lines = []
                target_lines = []
                if with_confidence: # binary input with OCR confidence?
                    sourceconf_lines = []
                if (repeat and not with_confidence):
                    fd.seek(0) # read again
                for line_no, line in enumerate(fd):
                    if (isinstance(split, np.ndarray) and 
                        (split[line_no] < split_ratio) == train):
                        # data shared between training and validation: belongs to other generator, resp.
                        #print('skipping line %d in favour of other generator' % line_no)
                        continue
                    if with_confidence: # binary input with OCR confidence?
                        source_conf, target_text = line # already includes end-of-sequence
                        source_text = u''.join([char for char, prob in source_conf])
                    else:
                        source_text, target_text = line.split(b'\t')
                         # add end-of-sequence:
                        source_text = source_text.decode('utf-8', 'strict') + u'\n'
                         # start-of-sequence will be added window by window,
                         # end-of-sequence already preserved by file iterator:
                        target_text = target_text.decode('utf-8', 'strict')
                    
                    # byte-align source and target text line, shelve them into successive fixed-size windows
                    self.aligner.set_seqs(source_text, target_text)
                    if train and self.aligner.is_bad():
                        #print('ignoring bad line "%s"' % source_text+target_text)
                        continue # avoid training if OCR was too bad
                    alignment1 = self.aligner.get_best_alignment()
                    if with_confidence: # binary input with OCR confidence?
                        # multiplex confidences into string alignment result:
                        k = 0
                        for i, (source_char, target_char) in enumerate(alignment1):
                            conf = 0
                            if source_char != self.aligner.GAP_ELEMENT:
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
                                         strict=train, verbose=(epoch == 1)))
                    source_lines.append(source_windows)
                    target_lines.append(target_windows)
                    if with_confidence:
                        sourceconf_lines.append(sourceconf_windows)

                    if len(source_lines) == self.batch_size: # end of batch
                        yield (source_lines, target_lines,
                               sourceconf_lines if with_confidence else None)
                        source_lines = []
                        target_lines = []
                        if with_confidence: # binary input with OCR confidence?
                            sourceconf_lines = []
                epoch += 1
                if repeat:
                    yield False
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
        # Ensure that no window cuts through a Unicode codepoint, and 
        # source/target window does not become empty (avoid by moving
        # last characters from previous window).
        source_windows = [[]]
        target_windows = [[b'\t'[0]]]
        sourceconf_windows = [[]]
        i = 0
        j = 1
        try:
            for source_char, target_char in line:
                if with_confidence: # binary input with OCR confidence?
                    source_char, source_conf = source_char
                if source_char != self.aligner.GAP_ELEMENT:
                    source_bytes = source_char.encode('utf-8')
                else:
                    source_bytes = b''
                if target_char != self.aligner.GAP_ELEMENT:
                    target_bytes = target_char.encode('utf-8')
                else:
                    target_bytes = b''
                source_len = len(source_bytes)
                target_len = len(target_bytes)
                if (i+source_len > self.window_length -3 or
                    j+target_len > self.window_length*1+1 -3): # window already full?
                    # or source_char == u' ' or target_char == u' '
                    if (strict and i == 0 and
                        len(bytes(target_windows[-1]).decode('utf-8', 'strict').strip(u'—-. \t')) > 0):
                        # empty source window, and not just line art in target window?
                        raise Exception("target window does not fit the source window in alignment",
                                        line,
                                        list(map(lambda l: bytes(l).decode('utf-8'),source_windows)),
                                        list(map(lambda l: bytes(l).decode('utf-8'),target_windows)))
                    if (strict and j == 1 and
                        len(bytes(source_windows[-1]).decode('utf-8', 'strict').strip(u'—-. ')) > 0):
                        # empty target window, and not just line art in source window?
                        raise Exception("source window does not fit the target window in alignment",
                                        line,
                                        list(map(lambda l: bytes(l).decode('utf-8'),source_windows)),
                                        list(map(lambda l: bytes(l).decode('utf-8'),target_windows)))
                    if i > self.window_length:
                        raise Exception("source window too long", i, j,
                                        list(map(lambda l: bytes(l).decode('utf-8'),source_windows)),
                                        list(map(lambda l: bytes(l).decode('utf-8'),target_windows)))
                    if j > self.window_length*1+1:
                        raise Exception("target window too long", i, j,
                                        list(map(lambda l: bytes(l).decode('utf-8'),source_windows)),
                                        list(map(lambda l: bytes(l).decode('utf-8'),target_windows)))
                    # make new window
                    if (i > 0 and j > 1 and
                        (source_char == self.aligner.GAP_ELEMENT and u'—-. '.find(target_char) < 0 or
                         target_char == self.aligner.GAP_ELEMENT and u'—-. '.find(source_char) < 0)):
                        # move last char from both current windows to new ones
                        source_window_last_len = (
                            len(bytes(source_windows[-1]).decode('utf-8', 'strict')[-1].encode('utf-8')))
                        target_window_last_len = (
                            len(bytes(target_windows[-1]).decode('utf-8', 'strict')[-1].encode('utf-8')))
                        source_windows.append(source_windows[-1][-source_window_last_len:])
                        target_windows.append([b'\t'[0]] + target_windows[-1][-target_window_last_len:])
                        source_windows[-2] = source_windows[-2][:-source_window_last_len]
                        target_windows[-2] = target_windows[-2][:-target_window_last_len]
                        i = source_window_last_len
                        j = target_window_last_len + 1
                        if with_confidence: # binary input with OCR confidence?
                            sourceconf_windows.append(sourceconf_windows[-1][-source_window_last_len:])
                            sourceconf_windows[-2] = sourceconf_windows[-2][:-source_window_last_len]
                    else:
                        i = 0
                        j = 1
                        source_windows.append([])
                        target_windows.append([b'\t'[0]]) # add start-of-sequence (for this window)
                        if with_confidence: # binary input with OCR confidence?
                            sourceconf_windows.append([])
                if source_char != self.aligner.GAP_ELEMENT:
                    source_windows[-1].extend(source_bytes)
                    i += source_len
                    if with_confidence: # binary input with OCR confidence?
                        sourceconf_windows[-1].extend(source_len*[source_conf])
                if target_char != self.aligner.GAP_ELEMENT:
                    target_windows[-1].extend(target_bytes)
                    j += target_len
        except Exception as e:
            if verbose:
                print('\x1b[2K\x1b[G', end='') # erase (progress bar) line and go to start of line
                print('windowing error: ', end='')
                print(e)
            # rid of the offending window, but keep the previous ones:
            source_windows.pop()
            target_windows.pop()
            if with_confidence: # binary input with OCR confidence?
                sourceconf_windows.pop()
        return source_windows, target_windows, sourceconf_windows
    
    def configure(self, batch_size=None):
        '''Define and compile encoder and decoder models for the configured parameters.
        
        Use given `batch_size` for encoder input if stateful: 
        configure once for training phase (with parallel lines),
        then reconfigure for prediction (with only 1 line each).
        (Decoder input will always have `self.batch_size`, 
        either from parallel input lines during training phase, 
        or from parallel hypotheses during prediction.)
        '''
        from keras.layers import Input, Dense, TimeDistributed, Dropout
        from keras.layers import LSTM, CuDNNLSTM, Bidirectional, Lambda
        from keras.layers import concatenate, average, add
        from keras.models import Model
        from keras import backend as K
        import tensorflow as tf
        
        if not batch_size:
            batch_size= self.batch_size
        
        # self.sess = tf.Session()
        # K.set_session(self.sess)
        
        # automatically switch to CuDNNLSTM if CUDA GPU is available:
        has_cuda = K.backend() == 'tensorflow' and K.tensorflow_backend._get_available_gpus()
        print('using', 'GPU' if has_cuda else 'CPU', 'LSTM implementation to compile',
              'stateful' if self.stateful else 'stateless', 
              'model of depth', self.depth, 'width', self.width,
              'window length', self.window_length)
        if self.residual_connections:
            print('encoder and decoder LSTM outputs are added to inputs in all hidden layers (residual_connections)')
        if self.deep_bidirectional_encoder:
            print('encoder LSTM is bidirectional in all hidden layers, ' +
                  'with fw/bw cross-summation between layers (deep_bidirectional_encoder)')
        if self.bridge_dense:
            print('state transfer between encoder and decoder LSTM uses ' +
                  'non-linear Dense layer as bridge in all hidden layers (bridge_dense)')
        lstm = CuDNNLSTM if has_cuda else LSTM
        
        ### Define training phase model
        
        # Set up an input sequence and process it.
        if self.stateful:
            # batch_size = 1 # override does not work (re-configuration would break internal encoder updates)
            encoder_inputs = Input(batch_shape=(batch_size, self.window_length, self.num_encoder_tokens))
        else:
            encoder_inputs = Input(shape=(self.window_length, self.num_encoder_tokens))
        # Set up the encoder. We will discard encoder_outputs and only keep encoder_state_outputs.
        #dropout/recurrent_dropout does not seem to help (at least for small unidirectional encoder),
        #go_backwards helps for unidirectional encoder with ~0.1 smaller loss on validation set (and not any slower),
        # unless UTF-8 byte strings are used directly
        encoder_state_outputs = []
        
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
        
        for n in range(self.depth):
            args = {'name': 'encoder_lstm_%d' % (n+1),
                    'return_state': True,
                    'return_sequences': (n < self.depth-1),
                    'stateful': self.stateful}
            if not has_cuda:
                # instead of default 'hard_sigmoid' which deviates from CuDNNLSTM:
                args['recurrent_activation'] = 'sigmoid'
            layer = lstm(self.width, **args)
            if self.deep_bidirectional_encoder:
                encoder_outputs, fw_state_h, fw_state_c, bw_state_h, bw_state_c = (
                    Bidirectional(layer, name=layer.name)(
                        encoder_inputs if n == 0 else cross_sum(encoder_outputs)))
                # prepare for current layer decoder initial_state:
                state_h = average([fw_state_h, bw_state_h]) # concatenate
                state_c = average([fw_state_c, bw_state_c]) # concatenate
            else:
                if n == 0:
                    encoder_outputs, fw_state_h, fw_state_c, bw_state_h, bw_state_c = (
                        Bidirectional(layer, name=layer.name)(encoder_inputs))
                    # prepare for base layer decoder initial_state:
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
            encoder_outputs = Dropout(self.dropout)(encoder_outputs)
            if self.bridge_dense:
                state_h = Dense(self.width, activation='tanh')(state_h)
                state_c = Dense(self.width, activation='tanh')(state_c)
            encoder_state_outputs.extend([state_h, state_c])
        
        # Set up an input sequence and process it.
        if self.stateful:
            # shape inference would assume fixed batch size here as well
            # (but we need that to be flexible for prediction):
            decoder_inputs = Input(batch_shape=(None, None, self.num_decoder_tokens))
        else:
            decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        # Set up decoder to return full output sequences (so we can train in parallel),
        # to use encoder_state_outputs as initial state and return final states as well. 
        # We don't use those states in the training model, but we will use them 
        # for inference (see further below). 
        decoder_lstms = []
        for n in range(self.depth):
            args = {'name': 'decoder_lstm_%d' % (n+1), 'return_state': True, 'return_sequences': True}
            if not has_cuda:
                # instead of default 'hard_sigmoid' which deviates from CuDNNLSTM:
                args['recurrent_activation'] = 'sigmoid'
            layer = lstm(self.width, **args) # self.width*2 if n == 0 else 
            decoder_lstms.append(layer)
            decoder_outputs2, _, _ = layer(decoder_inputs if n == 0 else decoder_outputs,
                                           initial_state=encoder_state_outputs[2*n:2*n+2])
            # add residual connections:
            if n > 0 and self.residual_connections:
                decoder_outputs = add([decoder_outputs2, decoder_outputs])
            else:
                decoder_outputs = decoder_outputs2
            decoder_outputs = Dropout(self.dropout)(decoder_outputs)
        # for experimenting with global normalization in beam search
        # (gets worse if done just like that): 'sigmoid'
        decoder_dense = TimeDistributed(Dense(self.num_decoder_tokens, activation='softmax'))
        decoder_outputs = decoder_dense(decoder_outputs)

        if self.lm_loss:
            for n in range(self.depth):
                layer = decoder_lstms[n] # tied weights
                lm_outputs, _, _ = layer(decoder_inputs if n == 0 else lm_outputs)
            lm_outputs = decoder_dense(lm_outputs)
            
            decoder_outputs = [decoder_outputs, lm_outputs] # 2 outputs, 1 combined loss
        
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
            state_h_in = Input(shape=(self.width,)) # self.width*2 if n == 0 else 
            state_c_in = Input(shape=(self.width,)) # self.width*2 if n == 0 else 
            decoder_state_inputs.extend([state_h_in, state_c_in])
            layer = decoder_lstms[n]
            decoder_outputs, state_h_out, state_c_out = layer(
                decoder_inputs if n == 0 else decoder_outputs, 
                initial_state=decoder_state_inputs[2*n:2*n+2])
            decoder_state_outputs.extend([state_h_out, state_c_out])
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_state_inputs,
            [decoder_outputs] + decoder_state_outputs)
        
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
    
    def train(self, filename):
        '''train model on text file
        
        Pass the UTF-8 byte sequence of lines in `filename`, 
        paired into source and target and aligned into fixed-length windows,
        to the loop training model weights with stochastic gradient descent.
        The generator will open the file, looping over the complete set (epoch)
        as long as validation error does not increase in between (early stopping).
        Validate on a random fraction of lines automatically separated before.
        (Data are always split by line, regardless of stateless/stateful mode.)
        '''
        from keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN
        from keras_train import fit_generator_autosized, evaluate_generator_autosized
        
        # Run training
        earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1,
                                      mode='min', restore_best_weights=True)
        callbacks = [earlystopping, TerminateOnNaN(),
                     StopSignalCallback(signal.SIGINT)]
        if self.stateful: # reset states between batches of different lines/documents (controlled by generator)
            reset_cb = self.ResetStatesCallback(self.encoder_model)
            callbacks.append(reset_cb)
        else:
            reset_cb = None
            
        # todo: shuffle lines
        # count how many batches the generator would return per epoch
        with open(filename, 'rb') as f:
            num_lines = sum(1 for line in f)
        print(u'Training on "%s" with %d lines' % (filename, num_lines))
        split_rand = np.random.uniform(0, 1, (num_lines,)) # reserve split fraction at random line numbers
        history = fit_generator_autosized(
            self.encoder_decoder_model,
            self.gen_data(filename, split_rand, train=True, reset_cb=reset_cb),
            epochs=self.epochs,
            workers=1, # more than 1 would effectively increase epoch size
            use_multiprocessing=self.scheduled_sampling == None,
            validation_data=self.gen_data(filename, split_rand, train=False, reset_cb=reset_cb),
            verbose=1,
            callbacks=callbacks,
            validation_callbacks=[reset_cb] if self.stateful else None)
        
        if 'val_loss' in history.history:
            print('training finished with val_loss ', min(history.history['val_loss']))
            if np.isnan(history.history['loss'][-1]):
                # recover weights (which TerminateOnNaN prevented EarlyStopping from doing)
                self.model.set_weights(earlystopping.best_weights)
            self.status = 2
        else:
            self.logger.critical('training failed')
            self.status = 1

    def evaluate(self, filename):
        '''evaluate model on text file
        
        Pass the UTF-8 byte sequence of lines in `filename`, 
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
        for batch_no, batch in enumerate(self.gen_windows(filename, False)):
            source_lines, target_lines, sourceconf_lines = batch
            #bar.update(1)
            if self.stateful:
                # model controlled by caller (batch prediction)
                #print('resetting encoder for line', line_no, train)
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
                encoder_input_data, decoder_input_data, decoder_output_data, _ = (
                    self.vectorize_windows(
                        [line[i] if len(line)>i else b'' for line in source_lines],
                        [line[i] if len(line)>i else b'' for line in target_lines],
                        [line[i] if len(line)>i else [] for line in sourceconf_lines] \
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
                    if i>=len(source_lines[j]) or i>=len(target_lines[j]):
                        continue # avoid empty window (masked during training)
                    source_seq = encoder_input_data[j]
                    target_seq = decoder_input_data[j]
                    # Take one sequence (part of the training/validation set) and try decoding it
                    source_texts[j] += bytes(source_lines[j][i]).decode("utf-8", "strict")
                    target_texts[j] += bytes(target_lines[j][i]).lstrip(b'\t').decode("utf-8", "strict")
                    greedy_texts[j] += self.decode_sequence_greedy(
                        source_state=[layer[j:j+1] for layer in source_states]).decode("utf-8", "strict")
                    try: # query only 1-best
                        beamed_texts[j] += next(self.decode_sequence_beam(
                            source_state=[layer[j:j+1] for layer in source_states],
                            eol=(i + 1 >= len(source_lines[j])))).decode("utf-8", "strict")
                    except StopIteration:
                        print('no beam decoder result within processing limits for '
                              '"%s\t%s" window %d of %d' % \
                              (source_texts[j], target_texts[j], i+1, len(source_lines[j])))
            
            for j in range(self.batch_size):
                if not len(source_lines[j]) or not len(target_lines[j]):
                    # ignore (zero) remainder of partially filled last batch
                    continue
                line_no = batch_no * self.batch_size + j
                
                print('Source input              : ', source_texts[j].rstrip(u'\n'))
                print('Target output             : ', target_texts[j].rstrip(u'\n'))
                print('Target prediction (greedy): ', greedy_texts[j].rstrip(u'\n'))
                print('Target prediction (beamed): ', beamed_texts[j].rstrip(u'\n'))
                
                #metric = self.aligner.get_levenshtein_distance
                metric = self.aligner.get_adjusted_distance
                
                c_edits_ocr, c_total = metric(source_texts[j],target_texts[j])
                c_edits_greedy, _ = metric(greedy_texts[j],target_texts[j])
                c_edits_beamed, _ = metric(beamed_texts[j],target_texts[j])
                
                greedy_tokens = greedy_texts[j].split(" ")
                beamed_tokens = beamed_texts[j].split(" ")
                source_tokens = source_texts[j].split(" ")
                target_tokens = target_texts[j].split(" ")

                w_edits_ocr, w_total = metric(source_tokens,target_tokens)
                w_edits_greedy, _ = metric(greedy_tokens,target_tokens)
                w_edits_beamed, _ = metric(beamed_tokens,target_tokens)
                
                counts[0] += c_total
                counts[1] += c_edits_ocr
                counts[2] += c_edits_greedy
                counts[3] += c_edits_beamed
                counts[4] += w_total
                counts[5] += w_edits_ocr
                counts[6] += w_edits_greedy
                counts[7] += w_edits_beamed
        
        print("CER OCR: {}".format(counts[1] / counts[0]))
        print("CER greedy: {}".format(counts[2] / counts[0]))
        print("CER beamed: {}".format(counts[3] / counts[0]))
        print("WER OCR: {}".format(counts[5] / counts[4]))
        print("WER greedy: {}".format(counts[6] / counts[4]))
        print("WER beamed: {}".format(counts[7] / counts[4]))

    def save(self, filename):
        '''Save model weights and configuration parameters.

        Save configured model parameters into `filename`.
        (This preserves weights across CPU/GPU implementations or input shape configurations.)
        '''
        assert self.status > 1 # already trained
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
    
    def load_weights(self, filename):
        '''Load weights into the configured/compiled model.

        Load weights from `filename` into the compiled and configured model.
        (This preserves weights across CPU/GPU implementations or input shape configurations.)
        '''
        assert self.status > 0 # already compiled
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
            load_weights_from_hdf5_group_by_name(file, self.encoder_decoder_model.layers,
                                                 skip_mismatch=True, reshape=False)
        self.status = 1

    def decode_batch_greedy(self, encoder_input_data):
        '''Predict from one batch array source window without alternatives.
        
        Use encoder input window array `encoder_input_data` (in a full batch).
        Start decoder with start-of-sequence, then keep decoding until
        end-of-sequence is found or output window length is full.
        Decode by using the best predicted output byte as next input.
        Pass decoder initial/final state from byte to byte.
        '''
        
        states_value = self.encoder_model.predict_on_batch(encoder_input_data)
        decoder_output_data = np.zeros((self.batch_size, self.window_length*1+1, self.num_decoder_tokens), dtype=np.int8)
        decoder_input_data = np.zeros((self.batch_size, 1, self.num_decoder_tokens), dtype=np.int8)
        decoder_input_data[:, 0, b'\t'[0]] = 1
        decoders = list(map(codecs.getincrementaldecoder('utf-8'), ['strict'] * self.batch_size))
        for i in range(self.window_length*1+1):
            decoder_output_data[:, i] = decoder_input_data[:, -1]
            output = self.decoder_model.predict_on_batch([decoder_input_data] + states_value)
            output_scores = output[0]
            output_states = output[1:]
            # if sampling from the raw distribution, we could stop here
            for j in range(self.batch_size):
                extra = decoders[j].getstate()
                while True:
                    decoders[j].setstate(extra)
                    index = np.nanargmax(output_scores[j, -1, :])
                    try:
                        char = bytes([index])
                        decoders[j].decode(char)
                        decoder_input_data[j] = np.eye(1, self.num_decoder_tokens, index)
                        break
                    except UnicodeDecodeError:
                        output_scores[j, -1, index] = np.nan
            states_value = list(output_states)
        return decoder_output_data
    
    def decode_sequence_greedy(self, source_seq=None, source_state=None):
        '''Predict from one line vector source window without alternatives.
        
        Use encoder input window vector `source_seq` (in a batch of size 1).
        If `source_state` is given, bypass that step to protect the encoder state.
        Start decoder with start-of-sequence, then keep decoding until
        end-of-sequence is found or output window length is full.
        Decode by using the best predicted output byte as next input.
        Pass decoder initial/final state from byte to byte.
        '''
        # reset must be done at line break (by caller)
        #self.encoder_model.reset_states()
        #self.decoder_model.reset_states()
       
        # Encode the source as state vectors.
        states_value = source_state if source_state != None else \
            self.encoder_model.predict_on_batch(np.expand_dims(source_seq, axis=0))
        
        # Generate empty target sequence of length 1.
        #target_seq = np.zeros((1, 1))
        target_seq = np.zeros((1, 1, self.num_decoder_tokens), dtype=np.int8)
        # Populate the first character of target sequence with the start character.
        #target_seq[0, 0, target_token_index['\t']-1] = 1.
        target_seq[0, 0, b'\t'[0]] = 1.
        #target_seq[0, 0] = b'\t'[0]
        decoder = codecs.getincrementaldecoder('utf8')()
        decoder.decode(b'\t')
        
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        decoded_text = b''
        for i in range(1, self.window_length*1+1):
            output = self.decoder_model.predict_on_batch([target_seq] + states_value)
            output_scores = output[0]
            output_states = output[1:]
            
            # Sample a token
            extra = decoder.getstate()
            while True:
                decoder.setstate(extra)
                sampled_token_index = np.nanargmax(output_scores[0, -1, :])
                #sampled_char = reverse_target_char_index[sampled_token_index+1]
                try:
                    sampled_char = bytes([sampled_token_index])
                    decoder.decode(sampled_char)
                    decoded_text += sampled_char
                    break
                except UnicodeDecodeError:
                    output_scores[0, -1, sampled_token_index] = np.nan # repeat without this output
            
            # Exit condition: either hit max length or find stop character.
            if sampled_char == b'\n':
                break
          
            # Update the target sequence (of length 1).
            #target_seq = np.zeros((1, 1))
            target_seq = np.zeros((1, 1, self.num_decoder_tokens), dtype=np.int8)
            #target_seq[0, 0] = sampled_token_index+1
            target_seq[0, 0, sampled_token_index] = 1.
            # feeding back the softmax vector directly vastly deteriorates predictions
            # (because only identity vectors are presented during training)
            # but we should add beam search (keep n best overall paths, add k best predictions here)
            
            # Update states
            states_value = list(output_states)
        
        return decoded_text.rstrip(bytes([0])) # strip trailing but keep intermediate zero bytes
    
    def decode_sequence_beam(self, source_seq=None, source_state=None, eol=False):
        '''Predict from one line vector source window with alternatives.
        
        Use encoder input window vector `source_seq` (in a batch of size 1).
        If `source_state` is given, bypass that step to protect the encoder state.
        Start decoder with start-of-sequence, then keep decoding until
        end-of-sequence is found or output window length is full, repeatedly
        (but at most beam_width_out times or a maximum number of steps).
        Decode by using the best predicted output byte and several next-best
        alternatives (up to some degradation threshold) as next input, 
        ensuring UTF-8 sequence validity.
        Follow-up on the n-best overall candidates (estimated by accumulated
        score, normalized by length), i.e. do breadth-first search.
        Pass decoder initial/final state from byte to byte, 
        for each candidate respectively.
        '''
        from bisect import insort_left
        
        decoder = codecs.getincrementaldecoder('utf8')()
        decoder.decode(b'\t')
        
        # reset must be done at line break (by caller)
        #self.encoder_model.reset_states()
        #self.decoder_model.reset_states()
        next_fringe = [Node(parent=None,
                            state=source_state if source_state != None else \
                                self.encoder_model.predict_on_batch(np.expand_dims(source_seq, axis=0)), # layer list of state arrays
                            value=b'\t'[0], # start symbol byte
                            cost=0.0,
                            extras=decoder.getstate())]
        hypotheses = []
        
        # generator will raise StopIteration if hypotheses is still empty after loop
        MAX_BATCHES = self.window_length*3 # how many batches (i.e. byte-hypotheses) will be processed per window?
        #avgpos = 0
        for l in range(MAX_BATCHES):
            # try:
            #     next(n for n in next_fringe if all(np.array_equal(x,y) for x,y in zip(nonbeam_states[:l+1], [s.state for s in n.to_sequence()])))
            # except StopIteration:
            #     print('greedy result falls off beam search at pos', l, nonbeam_seq[:l+1])
            fringe = []
            while next_fringe:
                n = next_fringe.pop()
                if n.value == b'\n'[0]: # end-of-sequence symbol?
                    if eol: # in last window?
                        hypotheses.append(n)
                        #print('found new solution "%s"' % bytes([i for i in n.to_sequence_of_values()]).rstrip(bytes([0])).decode("utf-8", "ignore"))
                elif n.length == self.window_length*1+1: # window full?
                    if not eol: # not last window?
                        hypotheses.append(n)
                        #print('found new solution "%s"' % bytes([i for i in n.to_sequence_of_values()]).rstrip(bytes([0])).decode("utf-8", "ignore"))
                else: # normal step
                    fringe.append(n)
                    #print('added new hypothesis "%s"' % bytes([i for i in n.to_sequence_of_values()]).decode("utf-8", "ignore"))
                if len(fringe) >= self.batch_size:
                    break # enough for one batch
            if not fringe:
                break # will give StopIteration unless we have some results already
            
            # use fringe leaves as minibatch, but with only 1 timestep
            target_seq = np.expand_dims(np.eye(256, dtype=np.int8)[[n.value for n in fringe], :], axis=1) # add time dimension
            states_val = [np.vstack([n.state[layer] for n in fringe]) for layer in range(len(fringe[0].state))] # stack layers across batch
            output = self.decoder_model.predict_on_batch([target_seq] + states_val)
            scores_output = output[0][:,-1] # only last timestep
            scores_output_order = np.argsort(scores_output, axis=1) # still in reverse order (worst first)
            states_output = list(output[1:]) # from (layers) tuple
            for i, n in enumerate(fringe): # iterate over batch (1st dim)
                scores = scores_output[i,:]
                scores_order = scores_output_order[i,:]
                highest = scores[scores_order[-1]]
                beampos = 256 - np.searchsorted(scores[scores_order], highest-0.3) # variable beam width
                #beampos = self.beam_width # fixed beam width
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
                #avgpos += pos
            if len(next_fringe) > MAX_BATCHES * self.batch_size: # more than can ever be processed within limits?
                next_fringe = next_fringe[-MAX_BATCHES*self.batch_size:] # to save memory, keep only best
        
        #avgpos /= l*self.batch_size
        #print ('total beam iterations: %d, average beam width: %.1f, results: %d, best cost: %.1f, worst cost: %.1f' % (l, avgpos, len(hypotheses), hypotheses[0].cum_cost, hypotheses[-1].cum_cost))
        hypotheses.sort(key=lambda n: n.cum_cost)
        for hypothesis in hypotheses[:self.beam_width_out]:
            indices = hypothesis.to_sequence_of_values()
            byteseq = bytes(indices[1:]).rstrip(bytes([0])) # strip trailing but keep intermediate zero bytes
            yield byteseq
    
class StopSignalCallback(Callback):
    '''Keras callback for graceful interruption of training.

    Halts training prematurely at the end of the current batch
    when the given signal was received once. If the callback
    gets to receive the signal again, exits immediately.
    '''

    def __init__(self, sig=signal.SIGINT, logger=None):
        super(StopSignalCallback, self).__init__()
        self.received = False
        self.sig = sig
        self.logger = logger or logging.getLogger(__name__)
        def stopper(sig, frame):
            if sig == self.sig:
                if self.received: # called again?
                    self.logger.critical('interrupting')
                    exit(0)
                else:
                    self.logger.critical('stopping training')
                    self.received = True
        self.action = signal.signal(self.sig, stopper)

    def __del__(self):
        signal.signal(self.sig, self.action)

    def on_batch_end(self, batch, logs=None):
        if self.received:
            self.model.stop_training = True

class ResetStatesCallback(Callback):
    '''Keras callback for stateful models to reset state between files.

    Callback to be called by `fit_generator()` or even `evaluate_generator()`:
    do `model.reset_states()` whenever generator sees EOF (on_batch_begin with self.eof),
    and between training and validation (on_batch_end with batch>=steps_per_epoch-1).
    '''
    def __init__(self, callback_model):
        self.eof = False
        self.here = ''
        self.next = ''
        self.callback_model = callback_model # different than self.model set by set_model()

    def reset(self, where):
        self.eof = True
        self.next = where

    def on_batch_begin(self, batch, logs={}):
        if self.eof:
            #print('resetting model at batch', batch, 'for train:', self.params['do_validation'])
            # called between training files,
            # reset only encoder (training does not converge if applied to complete encoder-decoder)
            self.callback_model.reset_states()
            self.eof = False
            self.here = self.next

    def on_batch_end(self, batch, logs={}):
        if logs.get('loss') > 10:
            pass # print(u'huge loss in', self.here, u'at', batch)

class Node(object):
    def __init__(self, parent, state, value, cost, extras):
        super(Node, self).__init__()
        self.value = value # byte
        self.parent = parent # parent Node, None for root
        self.state = state # recurrent layer hidden state
        self.cum_cost = parent.cum_cost + cost if parent else 0 # e.g. -log(p) of sequence up to current node (including)
        self.length = 1 if parent is None else parent.length + 1
        self.extras = extras
        self._norm_cost = self.cum_cost * (s2s.window_length*1+1-self.length+1) if parent else 0
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

# training+validation and evaluation sets (csv files: tab-separated lines)
traindata_gt = '../../daten/dta19-reduced/traindata.gt-gt.txt' # GT-only (clean) source text (for pretraining)
traindata_gt_large = '../../daten/dta19-reduced/dta_komplett_2017-09-01.18xy.rand300k.gt-gt.txt'
#traindata_ocr = '../../daten/dta19-reduced/traindata.Fraktur4-gt.filtered.txt' # OCR-only (noisy) source text
traindata_ocr = '../../daten/dta19-reduced/traindata.Fraktur4-gt.pkl' # OCR-only (noisy) source text with confidence
#traindata_ocr = '../../daten/dta19-reduced/traindata.foo4-gt.filtered.txt' # OCR-only (noisy) source text
#traindata_ocr = '../../daten/dta19-reduced/traindata.foo4-gt.pkl' # OCR-only (noisy) source text with confidence
#traindata_ocr = '../../daten/dta19-reduced/traindata.deu-frak3-gt.filtered.txt' # OCR-only (noisy) source text
#traindata_ocr = '../../daten/dta19-reduced/traindata.deu-frak3-gt.pkl' # OCR-only (noisy) source text with confidence
#traindata_ocr = '../../daten/dta19-reduced/traindata.ocrofraktur-gt.filtered.txt' # OCR-only (noisy) source text
#traindata_ocr = '../../daten/dta19-reduced/traindata.ocrofraktur-gt.pkl' # OCR-only (noisy) source text with confidence
#traindata_ocr = '../../daten/dta19-reduced/traindata.ocrofraktur-jze-gt.filtered.txt' # OCR-only (noisy) source text
#traindata_ocr = '../../daten/dta19-reduced/traindata.ocrofraktur-jze-gt.pkl' # OCR-only (noisy) source text with confidence
#traindata_ocr = '../../daten/dta19-reduced/traindata.mixed-gt.pkl' # OCR-only (noisy) source text with confidence
#traindata_ocr = '../../daten/GT4HistOCR/corpus/dta19/flattened/traindata.Fraktur4-gt.pkl' # OCR-only (noisy) source text with confidence
#traindata_ocr = '../../daten/GT4HistOCR/corpus/dta19/flattened/traindata.deu-frak3-gt.pkl' # OCR-only (noisy) source text with confidence
#traindata_ocr = '../../daten/GT4HistOCR/corpus/dta19/flattened/traindata.foo4-gt.pkl' # OCR-only (noisy) source text with confidence
#traindata_ocr = '../../daten/GT4HistOCR/corpus/dta19/flattened/traindata.ocrofraktur-gt.pkl' # OCR-only (noisy) source text with confidence
#traindata_ocr = '../../daten/GT4HistOCR/corpus/dta19/flattened/traindata.ocrofraktur-jze-gt.pkl' # OCR-only (noisy) source text with confidence
#traindata_ocr = '../../daten/GT4HistOCR/corpus/dta19/flattened/traindata.mixed-gt.pkl' # OCR-only (noisy) source text with confidence
#testdata_ocr = '../../daten/dta19-reduced/testdata.Fraktur4-gt.filtered.txt' # OCR-only (noisy) source text
testdata_ocr = '../../daten/dta19-reduced/testdata.Fraktur4-gt.pkl' # OCR-only (noisy) source text with confidence
#testdata_ocr = '../../daten/dta19-reduced/testdata.foo4-gt.filtered.txt' # OCR-only (noisy) source text
#testdata_ocr = '../../daten/dta19-reduced/testdata.foo4-gt.pkl' # OCR-only (noisy) source text with confidence
#testdata_ocr = '../../daten/dta19-reduced/testdata.deu-frak3-gt.filtered.txt' # OCR-only (noisy) source text
#testdata_ocr = '../../daten/dta19-reduced/testdata.deu-frak3-gt.pkl' # OCR-only (noisy) source text with confidence
#testdata_ocr = '../../daten/dta19-reduced/testdata.ocrofraktur-gt.filtered.txt' # OCR-only (noisy) source text
#testdata_ocr = '../../daten/dta19-reduced/testdata.ocrofraktur-gt.pkl' # OCR-only (noisy) source text with confidence
#testdata_ocr = '../../daten/dta19-reduced/testdata.ocrofraktur-jze-gt.filtered.txt' # OCR-only (noisy) source text
#testdata_ocr = '../../daten/dta19-reduced/testdata.ocrofraktur-jze-gt.pkl' # OCR-only (noisy) source text with confidence
#testdata_ocr = '../../daten/GT4HistOCR/corpus/dta19/flattened/testdata.Fraktur4-gt.pkl' # OCR-only (noisy) source text with confidence
#testdata_ocr = '../../daten/GT4HistOCR/corpus/dta19/flattened/testdata.deu-frak3-gt.pkl' # OCR-only (noisy) source text with confidence
#testdata_ocr = '../../daten/GT4HistOCR/corpus/dta19/flattened/testdata.foo4-gt.pkl' # OCR-only (noisy) source text with confidence
#testdata_ocr = '../../daten/GT4HistOCR/corpus/dta19/flattened/testdata.ocrofraktur-gt.pkl' # OCR-only (noisy) source text with confidence
#testdata_ocr = '../../daten/GT4HistOCR/corpus/dta19/flattened/testdata.ocrofraktur-jze-gt.pkl' # OCR-only (noisy) source text with confidence

model = traindata_ocr.split("/")[-1].split(".")[1].split("-")[0]
model_filename = u's2s.%s.d%d.w%04d.adam.window%d.%s.fast.h5' % (model, s2s.depth, s2s.width, s2s.window_length, "stateful" if s2s.stateful else "stateless")
if len(sys.argv) > 1:
    model_filename = sys.argv[1]
if len(sys.argv) > 2:
    s2s.beam_width = int(sys.argv[2])
    s2s.beam_width_out = s2s.beam_width

if isfile(model_filename):
    print(u'Loading model', model_filename)
    s2s.load_config(model_filename)
    s2s.configure()
    s2s.load_weights(model_filename)
    
    s2s.evaluate(testdata_ocr)
else:
    s2s.configure()
    model_filename_pretrained = model_filename.find('.pretrained+')
    if model_filename_pretrained > 0:
        model_filename_pretrained = model_filename[0:model_filename_pretrained+11]+'.h5'
        if not isfile(model_filename_pretrained):
            model_filename_pretrained = None
    if model_filename_pretrained:
        print(u'Loading pretrained weights', model_filename_pretrained)
        s2s.load_weights(model_filename_pretrained)
    else:
        model_filename_shallow = model_filename.replace(".d%d." % s2s.depth, ".d%d." % (s2s.depth-1))
        model_filename_lm = "lm.dta18.d%d.w%04d.h5" % (s2s.depth, s2s.width)
        if s2s.depth > 1 and isfile(model_filename_shallow):
            print(u'Loading shallower model weights', model_filename_shallow)
            s2s.load_transfer_weights(model_filename_shallow) # will only find depth-1 weights
            for i in range(1, s2s.depth): # fix previous layer weights
                s2s.encoder_decoder_model.get_layer(name='encoder_lstm_%d'%i).trainable = False
                s2s.encoder_decoder_model.get_layer(name='decoder_lstm_%d'%i).trainable = False
            s2s.recompile() # necessary for trainable to take effect
        elif isfile(model_filename_lm):
            print(u'Loading LM model weights', model_filename_lm)
            s2s.load_transfer_weights(model_filename_lm) # will only find decoder=LM weights
        s2s.train(traindata_gt)
        #s2s.train(traindata_gt_large)
    # s2s.decoder_model.trainable = False # seems to have an adverse effect
    # s2s.recompile() # necessary for trainable to take effect
    # reset weights of pretrained encoder (i.e. keep only decoder weights as initialization):
    from keras import backend as K
    session = K.get_session()
    for layer in s2s.encoder_model.layers:
        for v in layer.__dict__:
            v_arg = getattr(layer,v)
            if hasattr(v_arg, 'initializer'):
                initializer_method = getattr(v_arg, 'initializer')
                initializer_method.run(session=session)
    
    s2s.train(traindata_ocr)
    
    # Save model
    print(u'Saving model', model_filename)
    s2s.save(model_filename)
    
    s2s.evaluate(testdata_ocr)


# reconfigure to make encoder accept only 1 line at a time for interactive testing
if s2s.stateful: # encoder stateful?
    pass # re-configuration to batch_size 1 does not work, would break internal encoder updates (extend to full batch_size instead below)
else:
    s2s.configure(batch_size=1)
    
def transcode_line(source_line):
    # source_sequence = np.zeros((1, max_encoder_seq_length, s2s.num_encoder_tokens), dtype='float32')
    # for t, char in enumerate(source_text):
    #     source_sequence[0, t, source_token_index[char]] = 1
    # return source_sequence
    #return to_categorical(pad_sequences(source_tokenizer.texts_to_sequences([source_text]), maxlen=max_encoder_seq_length, padding='pre'), num_classes=s2s.num_encoder_tokens+1)[:,:,1:] # remove separate dimension for zero/padding
    #return to_categorical(pad_sequences([list(map(bytearray,source_text))], maxlen=max_encoder_seq_length, padding='pre'), num_classes=s2s.num_encoder_tokens+1)[:,:,1:] # remove separate dimension for zero/padding
    #return to_categorical(list(map(bytearray,[source_text + b'\n'])), num_classes=s2s.num_encoder_tokens+1)[:,:,1:]
    #return bytearray(source_text+b'\n')
    #return np.eye(256, dtype=np.float32)[bytearray(source_text+b'\n'),:]
    s2s.encoder_model.reset_states() # new line
    source_windows = [[]]
    for source_char in source_line + u'\n':
        source_bytes = source_char.encode('utf-8')
        source_len = len(source_bytes)
        if len(source_windows[-1])+source_len >= s2s.window_length-3:
            source_windows[-1].extend([0]*(s2s.window_length-len(source_windows[-1]))) # so np.eye works for learned zero-padding
            source_windows.append([])
        source_windows[-1].extend(source_bytes)
    source_windows[-1].extend([0]*(s2s.window_length-len(source_windows[-1]))) # so np.eye works for learned zero-padding
    target_line = u''
    for source_window in source_windows:
        source_seq = np.expand_dims(np.eye(256, dtype=np.int8)[list(bytearray(source_window)),:], axis=0) # add batch dimension
        if s2s.stateful:
            source_seq = np.repeat(source_seq, s2s.batch_size, axis=0) # repeat to full batch size
        source_states = s2s.encoder_model.predict_on_batch(source_seq) # get encoder output
        source_state = [layer[0:1] for layer in source_states] # get layer list for only 1 line
        #target_line += next(s2s.decode_sequence_beamed(source_state=source_state, eol=(b'\n' in source_window))).decode('utf-8', 'strict')
        target_line += s2s.decode_sequence_greedy(source_state=source_state).decode('utf-8', 'strict')
    return target_line
        

print(u"usage example:\n# transcode_line(u'hello world!')\n# s2s.evaluate('%s')" % testdata_ocr)
