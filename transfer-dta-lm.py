#!/usr/bin/python -i
import shutil, h5py, sys
source = '../../tools/ocrd_keraslm/model_dta18_l2_512bs_320.weights.h5'
target = 'lm.dta18.d2.w0320.h5'
if len(sys.argv) > 1:
    if sys.argv[1] in ['-h', '-help', '--help']:
        print('usage: %s [source-file [target-file]]\n\ndefault source-file: %s\ndefault target-file: %s\n' %
              (sys.argv[0], source, target))
        exit()
    else:
        source = sys.argv[1]
    if len(sys.argv) > 2:
        target = sys.argv[2]

shutil.copy(source, target)

with h5py.File(target, 'r+') as f:
     # default name in ocrd_keraslm vs name used by s2s (weight-tied to LM)
    f.copy('cu_dnnlstm_1', 'decoder_lstm_1')
    f.copy('cu_dnnlstm_2', 'decoder_lstm_2')
    #f.copy('dense_1', 'time_distributed_1')
    del f['cu_dnnlstm_1']
    del f['cu_dnnlstm_2']
    #del f['dense_1']
    rename = {b'cu_dnnlstm_1': b'decoder_lstm_1', b'cu_dnnlstm_2': b'decoder_lstm_2'} #b'dense_1': b'time_distributed_1'}
    names = f.attrs['layer_names'].astype('|S18') # longer
    for i in range(names.shape[0]):
        names[i] = rename.get(names[i],names[i])
    #f.attrs.modify('layer_names', names)
    f.attrs['layer_names'] = names
    print(f.attrs['layer_names'])
    f.flush()




    
