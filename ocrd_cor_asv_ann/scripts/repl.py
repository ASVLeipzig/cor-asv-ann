# -*- coding: utf-8
import os
import logging
import code
import readline
import rlcompleter
import click

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
    if 'PYTHONSTARTUP' in os.environ:
        exec(open(os.environ['PYTHONSTARTUP']).read())
    if not 'TF_CPP_MIN_LOG_LEVEL' in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    logging.basicConfig(level=logging.DEBUG)
    
    s2s = Sequence2Sequence(logger=logging.getLogger(__name__), progbars=True)
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
        source_windows, _, _ = s2s.window_line(zip(source_line + '\n', source_line + '\n'), verbose=True)
        target_line = ''
        for source_window in source_windows:
            source_seq = s2s.batch_size * [source_window] # repeat to batch size
            source_seq, _, _ = s2s.vectorize_windows(source_seq, source_seq)
            source_states = s2s.encoder_model.predict_on_batch(source_seq) # get encoder output
            source_state = [layer[0:1] for layer in source_states] # get layer list for only 1 line
            #target_line += next(s2s.decode_sequence_beamed(source_state=source_state, eol=('\n' in source_window)))
            target_line += s2s.decode_sequence_greedy(source_state=source_state)
        return target_line
    print("usage example:\n"
          ">>> transcode_line('hello world!')\n"
          ">>> s2s.evaluate('filename')\n"
          "now entering REPL...\n")

    bindings = globals()
    bindings.update(locals())
    readline.set_completer(rlcompleter.Completer(bindings).complete)
    readline.parse_and_bind('tab:complete')
    code.interact(local=bindings)
