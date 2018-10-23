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
import numpy as np
# these should all be wrapped in functions:
import pickle, click
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
        self.batch_size = 64  # How many samples are trained together? (batch length)
        self.window_length = 7 # How many bytes are encoded at once? (sequence length)
        self.stateful = True # stateful encoder (implicit state transfer between batches)
        self.width = 320  # latent dimensionality of the encoding space (hidden layer state)
        self.depth = 4 # number of encoder/decoder layers stacked above each other (only 1st layer will be BLSTM)
        self.epochs = 100  # maximum number of epochs to train for (unless stopping early by validation loss)
        self.beam_width = 4 # keep track of how many alternative sequences during decode_sequence_beam()?
        self.beam_width_out = self.beam_width # up to how many results can be drawn from generator decode_sequence_beam()?
        
        self.num_encoder_tokens = 256 # now utf8 bytes (including nul for non-output, tab for start, newline for stop)
        self.num_decoder_tokens = 256 # now utf8 bytes (including nul for non-output, tab for start, newline for stop)
        
        self.encoder_decoder_model = None
        self.encoder_model = None
        self.decoder_model = None
        
        self.status = 0 # empty / configured / trained?
    
    # for fit_generator()/predict_generator()/evaluate_generator()/standalone -- looping, but not shuffling
    def gen_data(self, filename, batch_size=None, get_size=False, get_edits=False, split=None, train=False, reset_cb=None, tf_session=None, tf_graph=None):
        '''generate batches of vector data from text file
        
        Opens `filename` in binary mode, loops over it (unless `get_size` or `get_edits`),
        producing one window of `batch_size` lines at a time.
        Pads windows to a `self.window_length` multiple of the longest line, respectively.
        If stateful, calls `reset_cb` at the start of each line (if given) or resets model directly (otherwise).
        Skips lines at `split` positions, depending on `train` (upper vs lower partition).
        Yields:
        - accumulated (greedy+beam) prediction error metrics if `get_edits`,
        - number of batches if `get_size`,
        - vector data batches (for fit_generator/evaluate_generator) otherwise.
        '''
        # alignment for evaluation only...
        from editdistance import eval as edit_eval

        # alignment for windowing...
        ## python-alignment is impractical with long or heavily deviating sequences (see github issues 9, 10, 11):
        #import alignment.sequence
        #alignment.sequence.GAP_ELEMENT = 0 # override default
        #from alignment.sequence import Sequence, GAP_ELEMENT
        #from alignment.vocabulary import Vocabulary
        #from alignment.sequencealigner import SimpleScoring, StrictGlobalSequenceAligner
        # Levenshtein scoring:
        #scoring = SimpleScoring(2,-1) # match score, mismatch score
        #aligner = StrictGlobalSequenceAligner(scoring,-2) # gap score
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
        # scoring = OCRScoring()
        # aligner = StrictGlobalSequenceAligner(scoring,-1) # gap score
        
        ## difflib is optimised for visual comparisons (Ratcliff-Obershelp), not minimal distance (Levenshtein):
        from difflib import SequenceMatcher
        matcher = SequenceMatcher(isjunk=None, autojunk=False)
        
        ## edit_distance is impractical with long sequences, even if very similar (GT lines > 1000 characters, see github issue 6)
        # from edit_distance.code import SequenceMatcher # similar API to difflib.SequenceMatcher
        # def char_similar(a, b):
        #     return (a == b or (a,b) in table)
        # matcher = SequenceMatcher(test=char_similar)
        
        ## edlib does not work on Unicode (non-ASCII strings)
        # import edlib
        
        if not batch_size:
            if self.stateful:
                batch_size = self.batch_size # cannot reduce to 1 here (difference to training would break internal encoder updates)
            else:
                batch_size = self.batch_size

        split_ratio = 0.2
        remainder_pass = False
        was_pretrained = (self.status > 1)
        
        lock = threading.Lock()
        with open(filename, 'rb') as f:
            epoch = 0
            if filename.endswith('.pkl'):
                f = pickle.load(f) # read once
            while True:
                if not remainder_pass:
                    source_lines = []
                    target_lines = []
                    if filename.endswith('.pkl'): # binary input with OCR confidence?
                        sourceconf_lines = []
                    epoch += 1
                if not filename.endswith('.pkl'):
                    f.seek(0) # read again
                batch_no = 0
                for line_no, line in enumerate(f):
                    if isinstance(split, np.ndarray) and (split[line_no] < split_ratio) == train:
                        #print('skipping line %d in favour of other generator' % line_no)
                        continue # data shared between training and validation: belongs to other generator
                    if len(source_lines) == 0 and not get_size: # start of batch
                        if self.stateful:
                            if reset_cb: # model controlled by callbacks (training)
                                reset_cb.reset("lines %d-%d" % (line_no, line_no+batch_size)) # inform callback
                            elif get_edits: # model controlled by caller (batch prediction)
                                #print('resetting encoder for line', line_no, train)
                                #self.encoder_decoder_model.reset_states() # does not work for some reason (never returns, even if passing tf_session.as_default() and tf_graph.as_default())
                                self.encoder_model.reset_states()
                                #self.decoder_model.reset_states() # not stateful yet
                    if filename.endswith('.pkl'): # binary input with OCR confidence?
                        source_conf, target_text = line # already includes end-of-sequence
                        source_text = u''.join([char for char, prob in source_conf])
                    else:
                        source_text, target_text = line.split(b'\t')
                        source_text = source_text.decode('utf-8', 'strict') + u'\n' # add end-of-sequence
                        target_text = target_text.decode('utf-8', 'strict') # start-of-sequence will be added window by window, end-of-sequence already preserved by file iterator
                    
                    # byte-align source and target text line, shelve them into successive fixed-size windows
                    GAP_ELEMENT = 0
                    ## code for python_alignment:
                    #vocabulary = Vocabulary() # inefficient, but helps keep search space smaller if independent for each line
                    #source_seq = vocabulary.encodeSequence(Sequence(source_text))
                    #target_seq = vocabulary.encodeSequence(Sequence(target_text))
                    #score = aligner.align(source_seq, target_seq)
                    #if train and score < -10 and score < 5-len(source_text):
                    #    #print('ignoring line (%d/%d)):' % (score, len(source_text)))
                    #    #print(source_text, end='')
                    #    #print(target_text, end='')
                    #    continue # avoid training if OCR is too bad
                    #score, alignments = aligner.align(source_seq, target_seq, backtrace=True)
                    #alignment1 = vocabulary.decodeSequenceAlignment(alignments[0])
                    #print ('alignment score:', alignment1.score)
                    #print ('alignment rate:', alignment1.percentIdentity())

                    ## code for difflib/edit_distance:
                    # matcher = difflib_matcher if len(source_text) > 4000 or len(target_text) > 4000 else editdistance_matcher
                    matcher.set_seqs(source_text, target_text)
                    # if train and matcher.distance() > 10 and matcher.distance() > len(source_text)-5:
                    if train and matcher.quick_ratio() < 0.5 and len(source_text) > 5:
                        continue # avoid training if OCR was too bad
                    alignment1 = []
                    for op, source_begin, source_end, target_begin, target_end in matcher.get_opcodes():
                        if op == 'equal':
                            alignment1.extend(zip(source_text[source_begin:source_end], target_text[target_begin:target_end]))
                        elif op == 'replace': # not really substitution:
                            delta = source_end-source_begin-target_end+target_begin
                            #alignment1.extend(zip(source_text[source_begin:source_end] + [GAP_ELEMENT]*(-delta), target_text[target_begin:target_end] + [GAP_ELEMENT]*(delta)))
                            if delta > 0: # replace+delete
                                alignment1.extend(zip(source_text[source_begin:source_end-delta], target_text[target_begin:target_end]))
                                alignment1.extend(zip(source_text[source_end-delta:source_end], [GAP_ELEMENT]*(delta)))
                            if delta <= 0: # replace+insert
                                alignment1.extend(zip(source_text[source_begin:source_end], target_text[target_begin:target_end+delta]))
                                alignment1.extend(zip([GAP_ELEMENT]*(-delta), target_text[target_end+delta:target_end]))
                        elif op == 'insert':
                            alignment1.extend(zip([GAP_ELEMENT]*(target_end-target_begin), target_text[target_begin:target_end]))
                        elif op == 'delete':
                            alignment1.extend(zip(source_text[source_begin:source_end], [GAP_ELEMENT]*(source_end-source_begin)))
                        else:
                            raise Exception("difflib returned invalid opcode", op, "in", line)
                    assert source_end == len(source_text)
                    assert target_end == len(target_text)
                    ## code for edlib:
                    # edres = edlib.align(source_text, target_text, mode='NW', task='path', k=max(len(source_text),len(target_text))*2)
                    # if edres['editDistance'] < 0:
                    #     print(line)
                    #     continue
                    # alignment1 = []
                    # n = ""
                    # source_k = 0
                    # target_k = 0
                    # for c in edres['cigar']:
                    #     if c.isdigit():
                    #         n = n + c
                    #     else:
                    #         i = int(n)
                    #         n = ""
                    #         if c in "=X": # identity/substitution
                    #             alignment1.extend(zip(source_text[source_k:source_k+i], target_text[target_k:target_k+i]))
                    #             source_k += i
                    #             target_k += i
                    #         elif c == "I": # insert into target
                    #             alignment1.extend(zip(source_text[source_k:source_k+i], [GAP_ELEMENT]*i))
                    #             source_k += i
                    #         elif c == "D": # delete from target
                    #             alignment1.extend(zip([GAP_ELEMENT]*i, target_text[target_k:target_k+i]))
                    #             target_k += i
                    #         else:
                    #             raise Exception("edlib returned invalid CIGAR opcode", c)
                    # assert source_k == len(source_text)
                    # assert target_k == len(target_text)
                    
                    ## code for identity alignment (for GT-only training; faster, no memory overhead)
                    # alignment1 = zip(source_text, target_text)
                    
                    # Produce batches of training data:
                    # - fixed `batch_size` lines (each line post-padded to maximum window multiple among batch)
                    # - fixed `self.window_length` samples (each window post-padded to fixed sequence length)
                    # with decoder windows twice as long as encoder windows.
                    # Yield window by window when full, then inform callback `reset_cb`.
                    # Callback will do `reset_states()` on all stateful layers of model at `on_batch_begin`.
                    # If stateful, also throw away partial batches at the very end of the file.
                    # Ensure that no window cuts through a Unicode codepoint, and the target window fits
                    # within twice the source window (always move partial characters to the next window).
                    # todo:
                    # Try to always fit umlauts and certain other similar pairs in the same window,
                    # i.e. pay special attention if they have different byte lengths.
                    # Or is this precaution unnecessary when stateful?
                    # umlauts = {u"a": u"ä", u"o": u"ö", u"u": u"ü", u"A": u"Ä", u"O": u"Ä", u"U": u"Ü",
                    #            u"ä": u"aͤ", u"ö": u"oͤ", u"ü": u"uͤ", u"Ä": u"Aͤ", u"Ö": u"Oͤ", u"Ü": u"uͤ",
                    #            u'"': u"“", u"“": u"‘‘", u"‘": u"“",
                    #            u"'": u"’", u"’": u"”", u"”": u"’’",
                    #            u',': u'‚', u'‚': u'„', u'„': u'‚‚',
                    #            u'-': u'‐', u'‐': u'‐', u'‐': u'-', u'—': u'--', u'–': u'-',
                    #            u'=': u'⸗', u'⸗': u'⹀', u'⹀': u'=',
                    #            u'ſ': u'f', u'ß': u'ſs', u's': u'ſ', 
                    #            u'ã': u'ã', u'ẽ': u'ẽ', u'ĩ': u'ĩ', u'õ': u'õ', u'ñ': u'ñ', u'ũ': u'ũ', u'ṽ': u'ṽ', u'ỹ': u'ỹ',
                    #            # how about diacritical characters vs combining diacritical marks?
                    #            u'k': u'ft' }
                    try:
                        source_windows = [[]]
                        target_windows = [[b'\t'[0]]]
                        i = 0
                        j = 1
                        if filename.endswith('.pkl'): # binary input with OCR confidence?
                            sourceconf_windows = [[]]
                            k = 0
                        for source_char, target_char in alignment1:
                            source_bytes = source_char.encode('utf-8') if source_char != GAP_ELEMENT else b''
                            target_bytes = target_char.encode('utf-8') if target_char != GAP_ELEMENT else b''
                            source_len = len(source_bytes)
                            target_len = len(target_bytes)
                            if i+source_len > self.window_length -3 or j+target_len > self.window_length*1+1 -3: # window already full?
                                # or source_char == u' ' or target_char == u' '
                                if train and i == 0 and len(bytes(target_windows[-1]).decode('utf-8', 'strict').strip(u'—-. \t')) > 0: # empty source window, and not just line art in target window?
                                    raise Exception("target window does not fit twice the source window in alignment", alignment1, list(map(lambda l: bytes(l).decode('utf-8'),source_windows)), list(map(lambda l: bytes(l).decode('utf-8'),target_windows))) # line.decode("utf-8")
                                if train and j == 1 and len(bytes(source_windows[-1]).decode('utf-8', 'strict').strip(u'—-. ')) > 0: # empty target window, and not just line art in source window?
                                    raise Exception("source window does not fit half the target window in alignment", alignment1, list(map(lambda l: bytes(l).decode('utf-8'),source_windows)), list(map(lambda l: bytes(l).decode('utf-8'),target_windows))) # line.decode("utf-8")
                                if i > self.window_length:
                                    raise Exception("source window too long", i, j, list(map(lambda l: bytes(l).decode('utf-8'),source_windows)), list(map(lambda l: bytes(l).decode('utf-8'),target_windows))) # line.decode("utf-8")
                                if j > self.window_length*1+1:
                                    raise Exception("target window too long", i, j, list(map(lambda l: bytes(l).decode('utf-8'),source_windows)), list(map(lambda l: bytes(l).decode('utf-8'),target_windows))) # line.decode("utf-8")                                
                                # make new window
                                if (i > 0 and j > 1 and
                                    (source_char == GAP_ELEMENT and u'—-. '.find(target_char) < 0 or
                                     target_char == GAP_ELEMENT and u'—-. '.find(source_char) < 0)):
                                    # move last char from both current windows to new ones
                                    source_window_last_len = len(bytes(source_windows[-1]).decode('utf-8', 'strict')[-1].encode('utf-8'))
                                    target_window_last_len = len(bytes(target_windows[-1]).decode('utf-8', 'strict')[-1].encode('utf-8'))
                                    source_windows.append(source_windows[-1][-source_window_last_len:])
                                    target_windows.append([b'\t'[0]] + target_windows[-1][-target_window_last_len:])
                                    source_windows[-2] = source_windows[-2][:-source_window_last_len]
                                    target_windows[-2] = target_windows[-2][:-target_window_last_len]
                                    i = source_window_last_len
                                    j = target_window_last_len + 1
                                    if filename.endswith('.pkl'): # binary input with OCR confidence?
                                        sourceconf_windows.append(sourceconf_windows[-1][-source_window_last_len:])
                                        sourceconf_windows[-2] = sourceconf_windows[-2][:-source_window_last_len]
                                else:
                                    i = 0
                                    j = 1
                                    source_windows.append([])
                                    target_windows.append([b'\t'[0]]) # add start-of-sequence (for this window)
                                    if filename.endswith('.pkl'): # binary input with OCR confidence?
                                        sourceconf_windows.append([])
                            if source_char != GAP_ELEMENT:
                                source_windows[-1].extend(source_bytes)
                                i += source_len
                                if filename.endswith('.pkl'): # binary input with OCR confidence?
                                    assert source_char == source_conf[k][0], "characters from alignment and from confidence tuples out of sync at {} in \"{}\" \"{}\": {}".format(k, source_text, target_text, alignment1)
                                    sourceconf_windows[-1].extend(source_len*[source_conf[k][1]])
                                    k += 1
                            if target_char != GAP_ELEMENT:
                                target_windows[-1].extend(target_bytes)
                                j += target_len
                    except Exception as e:
                        if epoch == 1:
                            print('\x1b[2K\x1b[G', end='') # erase (progress bar) line and go to start of line
                            print('windowing error: ', end='')
                            print(e)
                        # rid of the offending window, but keep the previous ones:
                        source_windows.pop()
                        target_windows.pop()
                        if filename.endswith('.pkl'): # binary input with OCR confidence?
                            sourceconf_windows.pop()
                    source_lines.append(source_windows)
                    target_lines.append(target_windows)
                    if filename.endswith('.pkl'): # binary input with OCR confidence?
                        sourceconf_lines.append(sourceconf_windows)
                    
                    if len(source_lines) == batch_size: # end of batch
                        max_windows = max([len(line) for line in source_lines])                
                        if get_size: # merely calculate number of batches that would be generated?
                            batch_no += batch_size if get_edits else max_windows
                            source_lines = []
                            target_lines = []
                            continue
                        if get_edits:
                            source_texts = []
                            target_texts = []
                            decoded_texts = []
                            beamdecoded_texts = []
                        
                        # yield windows...
                        lock.acquire() # ensure no other generator instance interferes within the block of lines
                        for i in range(max_windows):
                            batch_no += 1
                            
                            # vectorize
                            encoder_input_sequences = list(map(bytearray,[line[i] if len(line)>i else b'' for line in source_lines]))
                            decoder_input_sequences = list(map(bytearray,[line[i] if len(line)>i else b'' for line in target_lines]))
                            if filename.endswith('.pkl'): # binary input with OCR confidence?
                                encoder_conf_sequences = [line[i] if len(line)>i else [] for line in sourceconf_lines]
                            # with windowing, we cannot use pad_sequences for zero-padding any more, 
                            # because zero bytes are a valid decoder input or output now (and different from zero input or output):
                            #encoder_input_sequences = pad_sequences(encoder_input_sequences, maxlen=self.window_length, padding='post')
                            #decoder_input_sequences = pad_sequences(decoder_input_sequences, maxlen=self.window_length*2, padding='post')
                            #encoder_input_data = np.eye(256, dtype=np.float32)[encoder_input_sequences,:]
                            #decoder_input_data = np.eye(256, dtype=np.float32)[decoder_input_sequences,:]
                            encoder_input_data = np.zeros((batch_size, self.window_length, self.num_encoder_tokens), dtype=np.float32 if filename.endswith('.pkl') else np.int8)
                            decoder_input_data = np.zeros((batch_size, self.window_length*1+1, self.num_decoder_tokens), dtype=np.int8)
                            for j, (enc_seq, dec_seq) in enumerate(zip(encoder_input_sequences, decoder_input_sequences)):
                                if not len(enc_seq):
                                    continue # empty window (i.e. j-th line is shorter than others): true zero-padding (masked)
                                else:
                                    # encoder uses 256 dimensions (including zero byte):
                                    encoder_input_data[j*np.ones(len(enc_seq), dtype=np.int8), 
                                                       np.arange(len(enc_seq), dtype=np.int8), 
                                                       np.array(enc_seq, dtype=np.int8)] = 1
                                    if filename.endswith('.pkl'): # binary input with OCR confidence?
                                        encoder_input_data[j*np.ones(len(enc_seq), dtype=np.int8), 
                                                           np.arange(len(enc_seq), dtype=np.int8), 
                                                           np.array(enc_seq, dtype=np.int8)] = np.array(encoder_conf_sequences[j], dtype=np.float32)
                                    # zero bytes in encoder input become 1 at zero-byte dimension: indexed zero-padding (learned)
                                    padlen = self.window_length - len(enc_seq)
                                    encoder_input_data[j*np.ones(padlen, dtype=np.int8), 
                                                       np.arange(len(enc_seq), self.window_length, dtype=np.int8), 
                                                       np.zeros(padlen, dtype=np.int8)] = 1
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
                            decoder_output_data = np.roll(decoder_input_data,-1,axis=1).astype(np.float32) # output data will be ahead by 1 timestep
                            decoder_output_data[:,-1,:] = np.zeros(self.num_decoder_tokens) # delete+pad start token rolled in at the other end
                            
                            # index of padded samples, so we can mask them with the sample_weight parameter during fit() below
                            decoder_output_weights = np.ones(decoder_output_data.shape[:-1], dtype=np.float32)
                            decoder_output_weights[np.all(decoder_output_data == 0, axis=2)] = 0.

                            if get_edits: # calculate edits from decoding
                                # avoid repeating encoder in both functions (greedy and beamed), because we can only reset at the first window
                                source_states = self.encoder_model.predict_on_batch(encoder_input_data)

                                for j in range(batch_size):
                                    if i == 0:
                                        source_texts.append(u'')
                                        target_texts.append(u'')
                                        decoded_texts.append(u'')
                                        beamdecoded_texts.append(u'')
                                    if i>=len(source_lines[j]) or i>=len(target_lines[j]):
                                        continue # avoid empty window (masked during training)
                                    source_seq = encoder_input_data[j]
                                    target_seq = decoder_input_data[j]
                                    # Take one sequence (part of the training/validation set) and try decoding it
                                    source_texts[j] += bytes(source_lines[j][i]).decode("utf-8", "strict")
                                    target_texts[j] += bytes(target_lines[j][i]).lstrip(b'\t').decode("utf-8", "strict")
                                    decoded_texts[j] += self.decode_sequence_greedy(source_state=[layer[j:j+1] for layer in source_states]).decode("utf-8", "ignore")
                                    try: # query only 1-best
                                        beamdecoded_texts[j] += next(self.decode_sequence_beam(source_state=[layer[j:j+1] for layer in source_states], eol=(i+1>=len(source_lines[j])))).decode("utf-8", "strict")
                                    except StopIteration:
                                        print('no beam decoder result within processing limits for', source_texts[j], target_texts[j], 'window', i+1, 'of', len(source_lines[j]))
                                        continue # skip this line's window
                            else:
                                if was_pretrained: # sample_weight quickly causes getting stuck with NaN in gradient updates and weights (regardless of loss function, optimizer, gradient clipping, CPU or GPU) when re-training
                                    yield ([encoder_input_data, decoder_input_data], decoder_output_data)
                                else:
                                    yield ([encoder_input_data, decoder_input_data], decoder_output_data, decoder_output_weights)
                                    
                        if get_edits:
                            for j in range(batch_size):
                                print('Source input  from', 'training:' if train else 'test:    ', source_texts[j].rstrip(u'\n'))
                                print('Target output from', 'training:' if train else 'test:    ', target_texts[j].rstrip(u'\n'))
                                print('Target prediction (greedy): ', decoded_texts[j].rstrip(u'\n'))
                                print('Target prediction (beamed): ', beamdecoded_texts[j].rstrip(u'\n'))
                                
                                c_total = len(target_texts[j])
                                edits = edit_eval(source_texts[j],target_texts[j])
                                c_edits_ocr = edits
                                edits = edit_eval(decoded_texts[j],target_texts[j])
                                c_edits_greedy = edits
                                edits = edit_eval(beamdecoded_texts[j],target_texts[j])
                                c_edits_beamed = edits
                                
                                decoded_tokens = decoded_texts[j].split(" ")
                                beamdecoded_tokens = beamdecoded_texts[j].split(" ")
                                source_tokens = source_texts[j].split(" ")
                                target_tokens = target_texts[j].split(" ")
                                
                                w_total = len(target_tokens)
                                edits = edit_eval(source_tokens,target_tokens)
                                w_edits_ocr = edits
                                edits = edit_eval(decoded_tokens,target_tokens)
                                w_edits_greedy = edits
                                edits = edit_eval(beamdecoded_tokens,target_tokens)
                                w_edits_beamed = edits
                                
                                yield (c_total, c_edits_ocr, c_edits_greedy, c_edits_beamed, w_total, w_edits_ocr, w_edits_greedy, w_edits_beamed)
                        
                        lock.release()
                        source_lines = []
                        target_lines = []
                        if filename.endswith('.pkl'): # binary input with OCR confidence?
                            sourceconf_lines = []
                
                if get_size: # return size, do not loop
                    yield batch_no
                    break
                elif get_edits: # do not loop
                    if source_lines and not remainder_pass: # except if partially filled batch remains
                        # re-enter loop once, adding just enough lines to complete batch
                        # but picking them randomly from the entire file:
                        assert train == False
                        split = np.random.uniform(0, 1, (line_no+1,))
                        split_ratio = (batch_size*1.5 - len(source_lines)) / (line_no+1) # slightly larger than exact ratio to ensure we will get enough lines (a little more does no harm)
                        print('wrapping around after %d lines to get %d more lines for last batch with a random split ratio of %.2f' % (line_no+1, batch_size-len(source_lines), split_ratio))
                        remainder_pass = True # throw away next time we get here
                    else:
                        break
                else:
                    yield False

    def configure(self, batch_size=None):
        '''Define and compile encoder and decoder models for the configured parameters.
        
        Use given `batch_size` for encoder input if stateful: 
        configure once for training phase (with parallel lines),
        then reconfigure for prediction (with only 1 line each).
        (Decoder input will always have `self.batch_size`, 
        either from parallel input lines during training phase, 
        or from parallel hypotheses during prediction.)
        '''
        from keras.layers import Input, Dense, TimeDistributed
        from keras.layers import LSTM, CuDNNLSTM, Bidirectional, concatenate
        from keras.models import Model
        from keras.optimizers import Adam
        from keras import backend as K
        
        if not batch_size:
            batch_size= self.batch_size
        
        # automatically switch to CuDNNLSTM if CUDA GPU is available:
        has_cuda = K.backend() == 'tensorflow' and K.tensorflow_backend._get_available_gpus()
        print('using', 'GPU' if has_cuda else 'CPU', 'LSTM implementation to compile',
              'stateful' if self.stateful else 'stateless', 
              'model of depth', self.depth, 'width', self.width,
              'window length', self.window_length)
        lstm = CuDNNLSTM if has_cuda else LSTM
        
        ### Define training phase model
        
        # Set up an input sequence and process it.
        if self.stateful:
            # batch_size = 1 # override does not work (re-configuration would break internal encoder updates)
            encoder_inputs = Input(batch_shape=(batch_size, self.window_length, self.num_encoder_tokens))
        else:
            encoder_inputs = Input(shape=(self.window_length, self.num_encoder_tokens))
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
        if self.stateful:
            decoder_inputs = Input(batch_shape=(None, None, self.num_decoder_tokens)) # shape inference would assume fixed batch size here as well (but we need that to be flexible for prediction)
        else:
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
        #decoder_dense = TimeDistributed(Dense(self.num_decoder_tokens, activation='sigmoid')) # for experimenting with global normalization in beam search (gets worse if done just like that)
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
        '''train model on text file
        
        Pass the UTF-8 byte sequence of lines in `filename`, 
        paired into source and target and aligned into fixed-length windows,
        to the loop training model weights with stochastic gradient descent.
        The generator will open the file, looping over the complete set (epoch)
        as long as validation error does not increase in between (early stopping).
        Validate on a random fraction of lines automatically separated before.
        (Data are always split by line, regardless of stateless/stateful mode.)
        '''
        from keras.callbacks import EarlyStopping, ModelCheckpoint
        from keras_train import fit_generator_autosized, evaluate_generator_autosized
        
        # Run training
        callbacks = [EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='min'),
                     ModelCheckpoint('s2s_last.h5', monitor='val_loss', # to be able to replay long epochs (crash/overfitting)
                                     save_best_only=True, save_weights_only=True, mode='min')]
        if self.stateful: # reset states between batches of different lines/documents (controlled by generator)
            reset_cb = self.ResetStatesCallback(self.encoder_model)
            callbacks.append(reset_cb)
        else:
            reset_cb = None
            
        # todo: shuffle lines
        # count how many batches the generator would return per epoch
        with open(filename, 'rb') as f:
            num_lines = sum(1 for line in f)
        split_rand = np.random.uniform(0, 1, (num_lines,)) # reserve split fraction at random line numbers
        fit_generator_autosized(self.encoder_decoder_model,
                                self.gen_data(filename, train=True, split=split_rand, reset_cb=reset_cb),
                                epochs=self.epochs,
                                workers=1, # more than 1 would effectively increase epoch size
                                use_multiprocessing=True,
                                validation_data=self.gen_data(filename, train=False, split=split_rand, reset_cb=reset_cb),
                                verbose=1,
                                callbacks=callbacks,
                                validation_callbacks=[reset_cb] if self.stateful else None)
        
        self.status = 2

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
        for c in self.gen_data(filename, get_edits=True): # get counts of results for batch_size lines at a time
            #bar.update(1)
            for i in range(8):
                counts[i] += c[i]

        print("CER OCR: {}".format(counts[1] / counts[0]))
        print("CER greedy: {}".format(counts[2] / counts[0]))
        print("CER beamed: {}".format(counts[3] / counts[0]))
        print("WER OCR: {}".format(counts[5] / counts[4]))
        print("WER greedy: {}".format(counts[6] / counts[4]))
        print("WER beamed: {}".format(counts[7] / counts[4]))

    def load_config(self, filename):
        '''Load parameters to prepare configuration/compilation.

        Load model configuration from `filename`.
        '''
        config = pickle.load(open(filename, mode='rb'))
        self.width = config['width']
        self.depth = config['depth']
        self.window_length = config['length'] if 'length' in config else 7 # old default
        self.stateful = config['stateful']
    
    def save_config(self, filename):
        '''Save parameters from configuration.

        Save configured model parameters into `filename`.
        '''
        assert self.status > 0 # already compiled
        config = {'width': self.width, 'depth': self.depth, 'length': self.window_length, 'stateful': self.stateful}
        pickle.dump(config, open(filename, mode='wb'))
    
    def load_weights(self, filename):
        '''Load weights into the configured/compiled model.

        Load weights from `filename` into the compiled and configured model.
        (This preserves weights across CPU/GPU implementations or input shape configurations.)
        '''
        assert self.status > 0 # already compiled
        self.encoder_decoder_model.load_weights(filename)
        self.status = 2
    
    def save_weights(self, filename):
        '''Save weights of the trained model.

        Save trained model weights into `filename`.
        (This preserves weights across CPU/GPU implementations or input shape configurations.)
        '''
        assert self.status > 1 # already trained
        self.encoder_decoder_model.save_weights(filename)
    
    def decode_sequence_greedy(self, source_seq=None, source_state=None):
        '''Predict from one source vector window without alternatives.
        
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
        states_value = source_state if source_state != None else self.encoder_model.predict_on_batch(np.expand_dims(source_seq, axis=0))
        
        # Generate empty target sequence of length 1.
        #target_seq = np.zeros((1, 1))
        target_seq = np.zeros((1, 1, self.num_decoder_tokens), dtype=np.int8)
        # Populate the first character of target sequence with the start character.
        #target_seq[0, 0, target_token_index['\t']-1] = 1.
        target_seq[0, 0, b'\t'[0]] = 1.
        #target_seq[0, 0] = b'\t'[0]
        
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        decoded_text = b''
        for i in range(1, self.window_length*1):
            output = self.decoder_model.predict_on_batch([target_seq] + states_value)
            output_scores = output[0]
            output_states = output[1:]
            
            # Sample a token
            sampled_token_index = np.argmax(output_scores[0, -1, :])
            #sampled_char = reverse_target_char_index[sampled_token_index+1]
            sampled_char = bytes([sampled_token_index])
            decoded_text += sampled_char
            
            # Exit condition: either hit max length or find stop character.
            if sampled_char == b'\n':
                break
          
            # Update the target sequence (of length 1).
            #target_seq = np.zeros((1, 1))
            target_seq = np.zeros((1, 1, self.num_decoder_tokens), dtype=np.int8)
            #target_seq[0, 0] = sampled_token_index+1
            target_seq[0, 0, sampled_token_index] = 1.
            # feeding back the softmax vector directly vastly deteriorates predictions (because only identity vectors are presented during training)
            # but we should add beam search (keep n best overall paths, add k best predictions here)
            
            # Update states
            states_value = list(output_states)
        
        return decoded_text.strip(bytes([0])) # strip trailing but keep intermediate zero bytes
    
    def decode_sequence_beam(self, source_seq=None, source_state=None, eol=False):
        '''Predict from one source vector window with alternatives.
        
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
                            state=source_state if source_state != None else self.encoder_model.predict_on_batch(np.expand_dims(source_seq, axis=0)), # layer list of state arrays
                            value=b'\t'[0], # start symbol byte
                            cost=0.0,
                            extras=decoder.getstate())]
        hypotheses = []
        
        # generator will raise StopIteration if hypotheses is still empty after loop
        MAX_BATCHES = self.window_length*3 # how many batches (i.e. byte-hypotheses) will be processed per window?
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
            if len(hypotheses) >= self.beam_width_out:
                break # done
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
                beampos = 256 - np.searchsorted(scores[scores_order], 0.1 * highest) # variable beam width
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
            if len(next_fringe) > MAX_BATCHES * self.batch_size: # more than can ever be processed within limits?
                next_fringe = next_fringe[-MAX_BATCHES*self.batch_size:] # to save memory, keep only best
        
        hypotheses.sort(key=lambda n: n.cum_cost)
        for hypothesis in hypotheses[:self.beam_width_out]:
            indices = hypothesis.to_sequence_of_values()
            byteseq = bytes(indices[1:]).strip(bytes([0])) # strip trailing but keep intermediate zero bytes
            yield byteseq
    
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
                # between training files
                self.callback_model.reset_states() # reset only encoder (training does not converge if applied to complete encoder-decoder)
                self.eof = False
                self.here = self.next
        
        def on_batch_end(self, batch, logs={}):
            if logs.get('loss') > 10:
                print('huge loss in', self.here, 'at', batch)


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

# training+validation and evaluation sets (csv files: tab-separated lines)
traindata_gt = '../../daten/dta19-reduced/traindata.gt-gt.txt' # GT-only (clean) source text (for pretraining)
traindata_gt_large = '../../daten/dta19-reduced/dta_komplett_2017-09-01.18xy.rand300k.gt-gt.txt'
#traindata_ocr = '../../daten/dta19-reduced/traindata.Fraktur4-gt.filtered.txt' # OCR-only (noisy) source text
traindata_ocr = '../../daten/dta19-reduced/traindata.Fraktur4-gt.pkl' # OCR-only (noisy) source text
#traindata_ocr = '../../daten/dta19-reduced/traindata.foo4-gt.filtered.txt' # OCR-only (noisy) source text
#traindata_ocr = '../../daten/dta19-reduced/traindata.deu-frak3-gt.filtered.txt' # OCR-only (noisy) source text
#traindata_ocr = '../../daten/dta19-reduced/traindata.ocrofraktur-gt.filtered.txt' # OCR-only (noisy) source text
#traindata_ocr = '../../daten/dta19-reduced/traindata.ocrofraktur-jze-gt.filtered.txt' # OCR-only (noisy) source text
#testdata_ocr = '../../daten/dta19-reduced/testdata.Fraktur4-gt.filtered.txt' # OCR-only (noisy) source text
testdata_ocr = '../../daten/dta19-reduced/testdata.Fraktur4-gt.pkl' # OCR-only (noisy) source text
#testdata_ocr = '../../daten/dta19-reduced/testdata.foo4-gt.filtered.txt' # OCR-only (noisy) source text
#testdata_ocr = '../../daten/dta19-reduced/testdata.deu-frak3-gt.filtered.txt' # OCR-only (noisy) source text
#testdata_ocr = '../../daten/dta19-reduced/testdata.ocrofraktur-gt.filtered.txt' # OCR-only (noisy) source text
#testdata_ocr = '../../daten/dta19-reduced/testdata.ocrofraktur-jze-gt.filtered.txt' # OCR-only (noisy) source text

model = traindata_ocr.split("/")[-1].split(".")[1].split("-")[0]
model_filename = u's2s.%s.d%d.w%04d.adam.window%d.%s.fast.h5' % (model, s2s.depth, s2s.width, s2s.window_length, "stateful" if s2s.stateful else "stateless")
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
    # s2s.train(traindata_augmented)
    s2s.train(traindata_gt)
    s2s.decoder_model.trainable = False # seems to have no effect
    s2s.train(traindata_ocr)
    
    # Save model
    print('Saving model', model_filename, config_filename)
    s2s.save_config(config_filename)
    if isfile('s2s_last.h5'):
        rename('s2s_last.h5', model_filename)
    else:
        s2s.save_weights(model_filename)
    
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
        if len(source_windows[-1])+source_len >= s2s.window_length:
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
        target_line += s2s.decode_sequence_greedy(source_state=source_state).decode('utf-8', 'ignore')
    return target_line
        

print("usage example:\n# transcode_line(u'hello world!')\n# s2s.evaluate('%s')" % testdata_ocr)
