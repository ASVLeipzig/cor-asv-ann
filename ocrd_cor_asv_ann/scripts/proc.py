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
@click.option('-C', '--charmap', default={}, help='mapping for input characters before passing to correction; ' \
              'can be used to adapt to character set mismatch between input and model (without relying on underspecification alone)')
@click.option('-S', '--old-suffix', default='', help='Suffix to remove from input files for output files')
@click.option('-s', '--new-suffix', default='.cor.txt', help='Suffix to append to input files for output files')
@click.argument('data', nargs=-1, type=click.Path(dir_okay=False, exists=True))
def cli(load_model, fast, rejection, charmap, old_suffix, new_suffix, data):
    """Apply a correction model on GT or text files.
    
    Load a sequence-to-sequence model from the given path.
    
    Then open the `data` files, (ignoring target side strings, if any)
    and apply the model to its (source side) strings in batches, accounting
    for input file names line by line.
    
    \b
    (Supported file formats are:
     - * (plain-text), with source lines,
     - * (tab-separated values), with source-target lines,
     - *.pkl (pickle dumps), with source-target lines, where source is either
       - a single string, or
       - a sequence of character-probability tuples.)
    
    For each input file, open a new output file derived from its file name
    by removing `old_suffix` (or the last extension) and appending `new_suffix`.
    Write the resulting lines to that output file.
    """
    if not 'TF_CPP_MIN_LOG_LEVEL' in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(message)s',
                        datefmt='%H:%M:%S')
    logger = logging.getLogger(__name__).setLevel(logging.INFO)
    
    s2s = Sequence2Sequence(logger=logging.getLogger(__name__), progbars=True)
    s2s.load_config(load_model)
    s2s.configure()
    s2s.load_weights(load_model)
    s2s.rejection_threshold = rejection
    
    outfile = None
    lastname = ''
    done = []
    logging.info("running on %d files", len(data))
    for filenames, lines, scores in s2s.predict(data, fast=fast, greedy=fast, charmap=charmap):
        for filename, line, score in zip(filenames, lines, scores):
            if lastname != filename:
                if outfile and not outfile.closed:
                    done.append(lastname)
                    outfile.close()
                if not filename:
                    logging.info("done with %d files", len(done))
                    break
                lastname = filename
                if old_suffix and old_suffix in filename:
                    basename = filename.replace(old_suffix, "")
                else:
                    basename, ext = os.path.splitext(filename)
                    if old_suffix:
                        logging.warning("input file '%s' does not contain suffix '%s', removing '%s'",
                                       filename, old_suffix, ext)
                if filename == basename:
                    logging.warning("input file '%s' does not have a suffix", filename)
                logging.info("writing to next output file '%s'", basename + new_suffix)
                outfile = open(basename + new_suffix, 'w', encoding='UTF-8')
            outfile.write(line)
    if outfile and not outfile.closed:
        outfile.close()
