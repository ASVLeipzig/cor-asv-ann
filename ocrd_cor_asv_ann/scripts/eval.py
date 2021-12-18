# -*- coding: utf-8
import os
import logging
import click

from ..lib.seq2seq import Sequence2Sequence

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-m', '--load-model', default="model.h5", help='model file to load',
              type=click.Path(dir_okay=False, exists=True))
# click.File is impossible since we do not now a priori whether
# we have to deal with pickle dumps (mode 'rb', includes confidence)
# or plain text files (mode 'r')
@click.option('-f', '--fast', is_flag=True, help='only decode greedily')
@click.option('-r', '--rejection', default=0.5, type=click.FloatRange(0, 1.0),
              help='probability of the input characters in all hypotheses (set 0 to use raw predictions)')
@click.option('-n', '--normalization', default='historic_latin', type=click.Choice(
    ["Levenshtein", "NFC", "NFKC", "historic_latin"]),
              help='normalize character sequences before alignment/comparison (set Levenshtein for none)')
@click.option('-C', '--charmap', default={}, help='mapping for input characters before passing to correction; ' \
              'can be used to adapt to character set mismatch between input and model (without relying on underspecification alone)')
@click.option('-l', '--gt-level', default=1, type=click.IntRange(1, 3),
              help='GT transcription level to use for historic_latin normlization (1: strongest, 3: none)')
@click.option('-c', '--confusion', default=10, type=click.IntRange(min=0),
              help='show this number of most frequent (non-identity) edits (set 0 for none)')
@click.option('-H', '--histogram', is_flag=True,
              help='aggregate and compare character histograms')
@click.argument('data', nargs=-1, type=click.Path(dir_okay=False, exists=True))
def cli(load_model, fast, rejection, normalization, charmap, gt_level, confusion, histogram, data):
    """Evaluate a correction model on GT files.
    
    Load a sequence-to-sequence model from the given path.
    
    Then apply on the file paths `data`, comparing predictions
    (both greedy and beamed) with GT target, and measuring
    error rates.
    
    \b
    (Supported file formats are:
     - * (tab-separated values), with source-target lines
     - *.pkl (pickle dumps), with source-target lines, where source is either
       - a single string, or
       - a sequence of character-probability tuples.)
    """
    if not 'TF_CPP_MIN_LOG_LEVEL' in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger(__name__).setLevel(logging.INFO)
    
    s2s = Sequence2Sequence(logger=logging.getLogger(__name__), progbars=True)
    s2s.load_config(load_model)
    s2s.configure()
    s2s.load_weights(load_model)
    s2s.rejection_threshold = rejection
    
    s2s.evaluate(data, fast, normalization, charmap, gt_level, confusion, histogram)
