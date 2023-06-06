# -*- coding: utf-8
import os
import logging
import click

from ..lib.seq2seq import Sequence2Sequence

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
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
@click.option('-v', '--valdata', multiple=True, help='file to use for validation (instead of random split)',
              type=click.Path(dir_okay=False, exists=True))
# click.File is impossible since we do not now a priori whether
# we have to deal with pickle dumps (mode 'rb', includes confidence)
# or plain text files (mode 'r')
@click.argument('data', nargs=-1, type=click.Path(dir_okay=False, exists=True))
def cli(save_model, load_model, init_model, reset_encoder, width, depth, valdata, data):
    """Train a correction model on GT files.
    
    Configure a sequence-to-sequence model with the given parameters.
    
    If given `load_model`, and its configuration matches the current parameters,
    then load its weights.
    If given `init_model`, then transfer its mapping and matching layer weights.
    (Also, if its configuration has 1 less hidden layers, then fixate the loaded
    weights afterwards.)
    If given `reset_encoder`, re-initialise the encoder weights afterwards.
    
    Then, regardless, train on the `data` files using early stopping.
    
    \b
    (Supported file formats are:
     - * (tab-separated values), with source-target lines
     - *.pkl (pickle dumps), with source-target lines, where source is either
       - a single string, or
       - a sequence of character-probability tuples.)
    
    If no `valdata` were given, split off a random fraction of lines for
    validation. Otherwise, use only those files for validation.
    
    If the training has been successful, save the model under `save_model`.
    """
    if not data:
        raise ValueError("Training needs at least one data file")
    
    if not 'TF_CPP_MIN_LOG_LEVEL' in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    
    s2s = Sequence2Sequence(logger=logging.getLogger(__name__), progbars=True)
    s2s.width = width
    s2s.depth = depth
    s2s.configure()
    
    # there could be both, a full pretrained model to load,
    # and a model to initialise parts from (e.g. only decoder for LM)
    if load_model:
        s2s.load_config(load_model)
        if s2s.width == width and s2s.depth == depth:
            logging.info('loading weights from existing model for incremental training')
            s2s.configure()
            s2s.load_weights(load_model)
        else:
            logging.warning('ignoring existing model due to different topology (width=%d, depth=%d)',
                            s2s.width, s2s.depth)
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
    
    s2s.train(data, valdata or None)
    if s2s.status > 1:
        s2s.save(save_model)
    
