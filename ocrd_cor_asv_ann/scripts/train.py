# -*- coding: utf-8
import os
import logging
import click

from ..lib.seq2seq import Sequence2Sequence

@click.command()
@click.option('-m', '--save-model', default="model.h5", help='model file for saving',
              type=click.Path(dir_okay=False, writable=True))
@click.option('--load-model', help='model file for loading (incremental/pre-training)',
              type=click.Path(dir_okay=False, exists=True))
@click.option('--init-model', help='model file for initialisation (transfer from LM or shallower model)',
              type=click.Path(dir_okay=False, exists=True))
@click.option('--reset-encoder', is_flag=True, help='reset encoder weights after load/init')
@click.option('-w', '--width', default=128, help='number of nodes per hidden layer',
              type=click.IntRange(min=1, max=9128))
@click.option('-d', '--depth', default=2, help='number of stacked hidden layers',
              type=click.IntRange(min=1, max=10))
@click.option('-l', '--length', default=7, help='number characters per batch (window size)',
              type=click.IntRange(min=1, max=1024))
# click.File is impossible since we do not now a priori whether
# we have to deal with pickle dumps (mode 'rb', includes confidence)
# or plain text files (mode 'r')
@click.argument('data', nargs=-1, type=click.Path(dir_okay=False, exists=True))
def cli(save_model, load_model, init_model, reset_encoder, width, depth, length, data):
    """Train a correction model.
    
    Configure a sequence-to-sequence model with the given parameters.
    
    If given `load_model`, and its configuration matches the current parameters,
    then load its weights.
    If given `init_model`, then transfer its mapping and matching layer weights.
    (Also, if its configuration has 1 less hidden layers, then fixate the loaded
    weights afterwards.)
    If given `reset_encoder`, re-initialise the encoder weights afterwards.
    
    Then, regardless, train on the file paths `data` using early stopping.
    
    If the training has been successful, save the model under `save_model`.
    """
    if not 'TF_CPP_MIN_LOG_LEVEL' in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    logging.basicConfig(level=logging.DEBUG)
    
    s2s = Sequence2Sequence(logger=logging.getLogger(__name__), progbars=True)
    s2s.width = width
    s2s.depth = depth
    s2s.window_length = length
    s2s.configure()
    
    # there could be both, a full pretrained model to load,
    # and a model to initialise parts from (e.g. only decoder for LM)
    if load_model:
        s2s.load_config(load_model)
        if s2s.width == width and s2s.depth == depth and s2s.window_length == length:
            logging.info('loading weights from existing model for incremental training')
            s2s.configure()
            s2s.load_weights(load_model)
        else:
            logging.warning('ignoring existing model due to different topology (width=%d, depth=%d, length=%d)',
                            s2s.width, s2s.depth, s2s.window_length)
    if init_model:
        s2s.configure()
        s2s.load_transfer_weights(init_model)
    
    if reset_encoder:
        # reset weights of pretrained encoder (i.e. keep only decoder weights as initialization):
        from keras import backend as K
        session = K.get_session()
        for layer in s2s.encoder_model.layers:
            for var in layer.__dict__:
                var_arg = getattr(layer, var)
                if hasattr(var_arg, 'initializer'):
                    initializer_method = getattr(var_arg, 'initializer')
                    initializer_method.run(session=session)
    
    s2s.train(data)
    if s2s.status > 1:
        s2s.save(save_model)
    
