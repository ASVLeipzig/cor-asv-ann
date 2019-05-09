# -*- coding: utf-8
import os
import logging
import click

from ..lib.seq2seq import Sequence2Sequence

@click.command()
@click.option('-m', '--load-model', default="model.h5", help='model file to load',
              type=click.Path(dir_okay=False, exists=True))
# click.File is impossible since we do not now a priori whether
# we have to deal with pickle dumps (mode 'rb', includes confidence)
# or plain text files (mode 'r')
@click.option('--fast', is_flag=True, help='only decode greedily')
@click.argument('data', nargs=-1, type=click.Path(dir_okay=False, exists=True))
def cli(load_model, fast, data):
    """Evaluate a correction model.
    
    Load a sequence-to-sequence model from the given path.
    
    Then apply on the file paths `data`, comparing predictions
    (both greedy and beamed) with GT target, and measuring
    error rates.
    """
    if not 'TF_CPP_MIN_LOG_LEVEL' in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    logging.basicConfig()
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    
    s2s = Sequence2Sequence(logger=logging.getLogger(__name__), progbars=True)
    s2s.load_config(load_model)
    s2s.configure()
    s2s.load_weights(load_model)
    
    s2s.evaluate(data, fast)
