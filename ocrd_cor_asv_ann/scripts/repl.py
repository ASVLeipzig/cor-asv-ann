# -*- coding: utf-8
import os
import atexit
import logging
import code
import readline
import rlcompleter
import click
import numpy as np

from ..lib.seq2seq import Sequence2Sequence

@click.command()
def cli():
    """Try a correction model interactively.
    
    Import Sequence2Sequence, instantiate `s2s`,
    then enter REPL.
    Also, provide function `transcode_line`
    for single line correction.
    """
    # load pythonrc even with -i
    historyPath = os.path.expanduser("~/." + __name__ + "_history")
    if not 'TF_CPP_MIN_LOG_LEVEL' in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    logging.basicConfig()
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    
    s2s = Sequence2Sequence(logger=logging.getLogger(__name__), progbars=True)
    def transcode_line(source_line):
        from matplotlib import pyplot
        s2s.encoder_model.reset_states()
        encoder_input_data, _, _, _ = s2s.vectorize_lines([source_line + '\n'], [source_line + '\n'])
        target_line, score, alignments = s2s.decode_sequence_greedy(encoder_input_data[0])
        pyplot.imshow(np.squeeze(np.array(alignments)))
        pyplot.show()
        return target_line, score
    def unvectorize(data):
        lines = []
        for linedata in data:
            line = ''
            for stepdata in linedata:
                idx = np.nanargmax(stepdata)
                char = s2s.mapping[1][idx]
                line += char
                if char == '\n':
                    break
            lines.append(line)
        return lines
    def save_history(path=historyPath):
        readline.write_history_file(path)
    atexit.register(save_history)
    if os.path.exists(historyPath):
        readline.read_history_file(historyPath)
    
    print("usage example:\n"
          ">>> s2s.load_config('model')\n"
          ">>> s2s.configure()\n"
          ">>> s2s.load_weights('model')\n"
          ">>> s2s.evaluate(['filename'])\n\n"
          ">>> transcode_line('hello world!')\n"
          "now entering REPL...\n")
    
    # batch = next(s2s.gen_lines([filename], False))
    # eidata, didata, dodata, _ = s2s.vectorize_lines(*batch)
    # list(map(print, unvectorize(dodata)))
    # dopred = s2s.encoder_decoder_model.predict_on_batch([eidata, didata])
    # list(map(print, unvectorize(dopred)))
    # eopred = s2s.encoder_model.predict_on_batch(eidata)
    # dopred, _, _, _, _, _ = s2s.decoder_model.predict_on_batch([didata] + eopred
    # list(map(print, unvectorize(dopred)))
    # dopred = s2s.decode_batch_greedy(eidata)
    # list(map(print, unvectorize(dopred)))

    bindings = globals()
    bindings.update(locals())
    readline.set_completer(rlcompleter.Completer(bindings).complete)
    readline.parse_and_bind('tab:complete')
    code.interact(local=bindings)
