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

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
def cli():
    """Try a correction model interactively.
    
    Import Sequence2Sequence, instantiate `s2s`,
    then enter REPL.
    Also, provide function `transcode_line`
    for single line correction.
    """
    # load pythonrc even with -i
    history_path = os.path.expanduser("~/." + __name__ + "_history")
    if not 'TF_CPP_MIN_LOG_LEVEL' in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    
    s2s = Sequence2Sequence(logger=logging.getLogger(__name__), progbars=True)
    def transcode_line(source_line):
        from matplotlib import pyplot, gridspec, font_manager
        class Formatter(object):
            def __init__(self, ax):
                self.ax = ax
            def __call__(self, x, y):
                xlocs = self.ax.xaxis.get_ticklocs()
                ylocs = self.ax.yaxis.get_ticklocs()
                xlabels = self.ax.xaxis.get_ticklabels()
                ylabels = self.ax.yaxis.get_ticklabels()
                xloc = np.searchsorted(xlocs, x)
                yloc = np.searchsorted(ylocs, y)
                try:
                    xlabel = xlabels[xloc].get_text()
                except IndexError:
                    xlabel = 'unknown'
                try:
                    ylabel = ylabels[yloc].get_text()
                except:
                    ylabel = 'unknown'
                return '%s|%s' % (xlabel, ylabel)
        encoder_input_data, _, _, _ = s2s.vectorize_lines([source_line + '\n'], [source_line + '\n'])
        gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1])
        prop = font_manager.FontProperties(family=[
            # we most likely need glyphs for historic Latin codepoints:
            'FreeMono',
            'STIXGeneral',
            'monospace'])
        target_line, prob, score, alignments = s2s.decode_sequence_greedy(encoder_input_data[0])
        ax1 = pyplot.subplot(gs[0])
        im1 = pyplot.imshow(np.squeeze(np.array(alignments)))
        pyplot.xticks(np.arange(len(source_line)), list(source_line), fontproperties=prop)
        pyplot.yticks(np.arange(len(target_line)), list(target_line), fontproperties=prop)
        ax1.yaxis.tick_right()
        ax1.format_coord = Formatter(ax1)
        pyplot.title('alignment')
        pyplot.colorbar(im1, ax=ax1)
        ax2 = pyplot.subplot(gs[1], sharey=ax1, xticks=[])
        im2 = pyplot.imshow(np.array(prob)[:, np.newaxis], cmap="plasma")
        pyplot.title('probs')
        pyplot.colorbar(im2, ax=ax2)
        pyplot.annotate('cor-asv-ann greedy (ppl=%.2f)' % np.exp(score), xy=(0.5, 0.98), xycoords='figure fraction')
        target_line, prob, score, alignments = next(s2s.decode_sequence_beam(encoder_input_data[0]))
        alignments = np.squeeze(np.array(alignments))
        ax3 = pyplot.subplot(gs[2], sharex=ax1)
        im3 = pyplot.imshow(np.where(alignments == 1.0, np.nan, alignments))
        im3.cmap.set_bad('red')
        pyplot.xticks(np.arange(len(source_line)), list(source_line), fontproperties=prop)
        pyplot.yticks(np.arange(len(target_line)), list(target_line), fontproperties=prop)
        ax3.yaxis.tick_right()
        ax3.format_coord = Formatter(ax3)
        pyplot.title('alignment')
        cb3 = pyplot.colorbar(im3, ax=ax3)
        #cb3.set_ticks(cb3.get_ticks() + [1.0])
        #cb3.ax.set_yticklabels([x.get_text() for x in cb3.ax.get_yticklabels()] + ['rejection'])
        cb3.ax.yaxis.get_ticklabels(which='both')[-1].set_text('rejection')
        ax4 = pyplot.subplot(gs[3], sharey=ax3, xticks=[])
        im4 = pyplot.imshow(np.array(prob)[:, np.newaxis], cmap="plasma")
        pyplot.title('probs')
        cb4 = pyplot.colorbar(im4, ax=ax4)
        #cb4.set_ticks(cb4.get_ticks() + [s2s.rejection_threshold])
        #cb4.ax.set_yticklabels([x.get_text() for x in cb4.ax.get_yticklabels(which='both')] + ['rejection_threshold'])
        cb4.ax.yaxis.get_ticklabels(which='both')[-1].set_text('rejection_threshold')
        pyplot.annotate('cor-asv-ann beamed (ppl=%.2f)' % np.exp(score), xy=(0.5, 0.48), xycoords='figure fraction')
        pyplot.show() # wait for user
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
    def save_history(path=history_path):
        readline.write_history_file(path)
    atexit.register(save_history)
    if os.path.exists(history_path):
        readline.read_history_file(history_path)
    
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
