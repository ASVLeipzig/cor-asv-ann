"""training/validation loops with autosized generators.

copied from keras/engine/training_generator.py, with modifications:
 - auto-sized generators (no steps_per_epoch): generator must
   yield False at epoch end, then wrap around
 - use progbar with target=None ("Unknown") during first epoch,
   re-use target=steps afterwards
 - callbacks during evaluation (e.g. for fine-grained reset)
 - progbar also during validation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import numpy as np

from keras import backend as K
from keras.utils.data_utils import Sequence
from keras.utils.data_utils import GeneratorEnqueuer
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.generic_utils import Progbar
from keras import callbacks as cbks


def fit_generator_autosized(model,
                  generator,
                  epochs=1,
                  #steps_per_epoch=None,
                  verbose=1,
                  callbacks=None,
                  validation_data=None,
                  validation_steps=None,
                  validation_callbacks=None,
                  class_weight=None,
                  max_queue_size=10,
                  workers=1,
                  use_multiprocessing=False,
                  shuffle=True,
                  initial_epoch=0):
    """See docstring for `Model.fit_generator`."""
    wait_time = 0.01  # in seconds
    epoch = initial_epoch

    do_validation = bool(validation_data)
    model._make_train_function()
    if do_validation:
        model._make_test_function()
    
    is_sequence = isinstance(generator, Sequence)
    if not is_sequence and use_multiprocessing and workers > 1:
        warnings.warn(
            UserWarning('Using a generator with `use_multiprocessing=True`'
                        ' and multiple workers may duplicate your data.'
                        ' Please consider using the`keras.utils.Sequence'
                        ' class.'))
    # if steps_per_epoch is None:
    #     if is_sequence:
    #         steps_per_epoch = len(generator)
    #     else:
    #         raise ValueError('`steps_per_epoch=None` is only valid for a'
    #                          ' generator based on the '
    #                          '`keras.utils.Sequence`'
    #                          ' class. Please specify `steps_per_epoch` '
    #                          'or use the `keras.utils.Sequence` class.')

    # python 2 has 'next', 3 has '__next__'
    # avoid any explicit version checks
    val_gen = (hasattr(validation_data, 'next') or
               hasattr(validation_data, '__next__') or
               isinstance(validation_data, Sequence))
    # if (val_gen and not isinstance(validation_data, Sequence) and
    #         not validation_steps):
    #     raise ValueError('`validation_steps=None` is only valid for a'
    #                      ' generator based on the `keras.utils.Sequence`'
    #                      ' class. Please specify `validation_steps` or use'
    #                      ' the `keras.utils.Sequence` class.')
    
    # Prepare display labels.
    out_labels = model.metrics_names
    callback_metrics = out_labels + ['val_' + n for n in out_labels]
    
    # prepare callbacks
    model.history = cbks.History()
    _callbacks = [cbks.BaseLogger(
        stateful_metrics=model.metrics_names[1:])]
    # instead of ProgbarLogger (but only for first epoch):
    if verbose:
        print('Epoch 1/%d' % epochs)
        progbar = Progbar(target=None, verbose=1, stateful_metrics=model.metrics_names[1:])
    _callbacks += (callbacks or []) + [model.history]
    callbacks = cbks.CallbackList(_callbacks)

    # it's possible to callback a different model than self:
    if hasattr(model, 'callback_model') and model.callback_model:
        callback_model = model.callback_model
    else:
        callback_model = model
    callbacks.set_model(callback_model)
    callbacks.set_params({
        'epochs': epochs,
        'steps': None, # will be refined during first epoch
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics,
    })
    callbacks.on_train_begin()
    
    enqueuer = None
    val_enqueuer = None
    
    try:
        if do_validation and not val_gen:
            # Prepare data for validation
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise ValueError('`validation_data` should be a tuple '
                                 '`(val_x, val_y, val_sample_weight)` '
                                 'or `(val_x, val_y)`. Found: ' +
                                 str(validation_data))
            val_x, val_y, val_sample_weights = model._standardize_user_data(
                val_x, val_y, val_sample_weight)
            val_data = val_x + val_y + val_sample_weights
            if model.uses_learning_phase and not isinstance(K.learning_phase(),
                                                            int):
                val_data += [0.]
            for cbk in callbacks:
                cbk.validation_data = val_data
        
        if workers > 0:
            if is_sequence:
                enqueuer = OrderedEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing,
                    shuffle=shuffle)
            else:
                enqueuer = GeneratorEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing,
                    wait_time=wait_time)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()
        else:
            if is_sequence:
                output_generator = iter(generator)
            else:
                output_generator = generator

        callback_model.stop_training = False
        # Construct epoch logs.
        epoch_logs = {}
        while epoch < epochs:
            model.reset_metrics()
            callbacks.on_epoch_begin(epoch)
            steps_done = 0
            batch_index = 0
            for generator_output in output_generator:
                if not generator_output: # end of epoch?
                    break
                if not hasattr(generator_output, '__len__'):
                    raise ValueError('Output of generator should be '
                                     'a tuple `(x, y, sample_weight)` '
                                     'or `(x, y)`. Found: ' +
                                     str(generator_output))

                if len(generator_output) == 2:
                    x, y = generator_output
                    sample_weight = None
                elif len(generator_output) == 3:
                    x, y, sample_weight = generator_output
                else:
                    raise ValueError('Output of generator should be '
                                     'a tuple `(x, y, sample_weight)` '
                                     'or `(x, y)`. Found: ' +
                                     str(generator_output))
                # build batch logs
                batch_logs = {}
                if not x:
                    # Handle data tensors support when no input given
                    # step-size = 1 for data tensors
                    batch_size = 1
                elif isinstance(x, list):
                    batch_size = x[0].shape[0]
                elif isinstance(x, dict):
                    batch_size = list(x.values())[0].shape[0]
                else:
                    batch_size = x.shape[0]
                batch_logs['batch'] = batch_index
                batch_logs['size'] = batch_size
                callbacks.on_batch_begin(batch_index, batch_logs)
                
                outs = model.train_on_batch(x, y,
                                            sample_weight=sample_weight,
                                            class_weight=class_weight,
                                            reset_metrics=False)
                
                if not isinstance(outs, list):
                    outs = [outs]
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o
                
                callbacks.on_batch_end(batch_index, batch_logs)
                if epoch == initial_epoch and verbose:
                    log_values = []
                    for k in callback_metrics:
                        if k in batch_logs:
                            log_values.append((k, batch_logs[k]))
                    progbar.update(steps_done, log_values)
                
                batch_index += 1
                steps_done += 1
                
                if callback_model.stop_training:
                    break

            if steps_done == 0:
                raise ValueError('Output of generator must not be empty')

            if epoch == initial_epoch:
                if verbose:
                    log_values = []
                    for k in callback_metrics:
                        if k in batch_logs:
                            log_values.append((k, batch_logs[k]))
                    progbar.update(steps_done, log_values)
            
            # Epoch finished.
            if do_validation:
                if val_gen:
                    val_outs, validation_steps = evaluate_generator_autosized(
                        model,
                        validation_data,
                        steps=validation_steps,
                        callbacks=validation_callbacks,
                        workers=workers,
                        use_multiprocessing=use_multiprocessing,
                        max_queue_size=max_queue_size,
                        verbose=1)
                else:
                    # No need for try/except because
                    # data has already been validated.
                    val_outs = model.evaluate(
                        val_x, val_y,
                        batch_size=batch_size,
                        sample_weight=val_sample_weights,
                        verbose=0)
                if not isinstance(val_outs, list):
                    val_outs = [val_outs]
                # Same labels assumed.
                for l, o in zip(out_labels, val_outs):
                    epoch_logs['val_' + l] = o
                
                if callback_model.stop_training:
                    break
            
            callbacks.on_epoch_end(epoch, epoch_logs)
            if epoch == initial_epoch:
                if verbose:
                    print()
                    progbar = cbks.ProgbarLogger(
                        count_mode='steps',
                        stateful_metrics=model.stateful_metric_names)
                    progbar.set_model(callback_model)
                    callbacks.append(progbar)
                callbacks.set_params({
                    'epochs': epochs,
                    'steps': steps_done, # refine
                    'verbose': verbose,
                    'do_validation': do_validation,
                    'metrics': callback_metrics,
                })
                if verbose:
                    progbar.on_train_begin()
            
            epoch += 1
            if callback_model.stop_training:
                break

            if is_sequence and workers == 0:
                generator.on_epoch_end()

    finally:
        try:
            if enqueuer is not None:
                enqueuer.stop()
        finally:
            if val_enqueuer is not None:
                val_enqueuer.stop()
    
    callbacks.on_train_end()
    return model.history

def evaluate_generator_autosized(model, generator,
                       steps=None,
                       callbacks=None,
                       max_queue_size=10,
                       workers=1,
                       use_multiprocessing=False,
                       verbose=0):
    """See docstring for `Model.evaluate_generator`."""
    model._make_test_function()
    
    model.reset_metrics()
    callbacks = cbks.CallbackList(callbacks or [])
    
    # it's possible to callback a different model than self:
    if hasattr(model, 'callback_model') and model.callback_model:
        callback_model = model.callback_model
    else:
        callback_model = model
    callbacks.set_model(callback_model)
    callbacks.set_params({
        'epochs': 1,
        'steps': steps, # if None, will be refined during first epoch
        'verbose': verbose,
        'do_validation': False,
        'metrics': model.metrics_names,
    })
        
    steps_done = 0
    wait_time = 0.01
    outs_per_batch = []
    batch_sizes = []
    is_sequence = isinstance(generator, Sequence)
    if not is_sequence and use_multiprocessing and workers > 1:
        warnings.warn(
            UserWarning('Using a generator with `use_multiprocessing=True`'
                        ' and multiple workers may duplicate your data.'
                        ' Please consider using the`keras.utils.Sequence'
                        ' class.'))
    # if steps is None:
    #     if is_sequence:
    #         steps = len(generator)
    #     else:
    #         raise ValueError('`steps=None` is only valid for a generator'
    #                          ' based on the `keras.utils.Sequence` class.'
    #                          ' Please specify `steps` or use the'
    #                          ' `keras.utils.Sequence` class.')
    enqueuer = None

    try:
        if workers > 0:
            if is_sequence:
                enqueuer = OrderedEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing)
            else:
                enqueuer = GeneratorEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing,
                    wait_time=wait_time)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()
        else:
            if is_sequence:
                output_generator = iter(generator)
            else:
                output_generator = generator

        if verbose == 1:
            progbar = Progbar(target=steps)
        callbacks.on_epoch_begin(0)

        for generator_output in output_generator:
            if not generator_output: # end of epoch?
                break
            if not hasattr(generator_output, '__len__'):
                raise ValueError('Output of generator should be a tuple '
                                 '(x, y, sample_weight) '
                                 'or (x, y). Found: ' +
                                 str(generator_output))
            if len(generator_output) == 2:
                x, y = generator_output
                sample_weight = None
            elif len(generator_output) == 3:
                x, y, sample_weight = generator_output
            else:
                raise ValueError('Output of generator should be a tuple '
                                 '(x, y, sample_weight) '
                                 'or (x, y). Found: ' +
                                 str(generator_output))
            # build batch logs
            batch_logs = {}
            if not x:
                # Handle data tensors support when no input given
                # step-size = 1 for data tensors
                batch_size = 1
            elif isinstance(x, list):
                batch_size = x[0].shape[0]
            elif isinstance(x, dict):
                batch_size = list(x.values())[0].shape[0]
            else:
                batch_size = x.shape[0]
            if batch_size == 0:
                raise ValueError('Received an empty batch. '
                                 'Batches should contain '
                                 'at least one item.')
            batch_logs['batch'] = steps_done
            batch_logs['size'] = batch_size
            callbacks.on_batch_begin(steps_done, batch_logs)
            
            outs = model.test_on_batch(x, y,
                                       sample_weight=sample_weight,
                                       reset_metrics=False)
            if not isinstance(outs, list):
                outs = [outs]
            for l, o in zip(model.metrics_names, outs):
                batch_logs[l] = o
            outs_per_batch.append(outs)
            
            callbacks.on_batch_end(steps_done, batch_logs)
            
            steps_done += 1
            batch_sizes.append(batch_size)
            if verbose == 1:
                log_values = []
                for k in model.metrics_names:
                    if k in batch_logs:
                        log_values.append(('val_' + k, batch_logs[k]))
                progbar.update(steps_done, log_values)

        callbacks.on_epoch_end(1, {})
        
    finally:
        if enqueuer is not None:
            enqueuer.stop()
    
    averages = [float(outs_per_batch[-1][0])]  # index 0 = 'loss'
    for i in range(1, len(outs)):
        averages.append(np.float64(outs_per_batch[-1][i]))
    if len(averages) == 1:
        return averages[0], steps_done
    return averages, steps_done
