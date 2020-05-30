# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Main training script."""

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import tpu
import tflex
import tqdm
import time
import dnnlib
import dnnlib.tflib as tflib
import traceback
from dnnlib.tflib.autosummary import autosummary, get_tpu_summary, set_num_replicas

from training import dataset
from training import misc
from metrics import metric_base
from pprint import pprint

#----------------------------------------------------------------------------
# Just-in-time processing of training images before feeding them to the networks.

def process_reals(x, labels, lod, mirror_augment, drange_data, drange_net):
    with tf.name_scope('DynamicRange'):
        x = tf.cast(x, tf.float32)
        x = misc.adjust_dynamic_range(x, drange_data, drange_net)
    if mirror_augment:
        with tf.name_scope('MirrorAugment'):
            x = tf.where(tf.random_uniform([tf.shape(x)[0]]) < 0.5, x, tf.reverse(x, [3]))
    with tf.name_scope('FadeLOD'): # Smooth crossfade between consecutive levels-of-detail.
        s = tf.shape(x)
        y = tf.reshape(x, [-1, s[1], s[2]//2, 2, s[3]//2, 2])
        y = tf.reduce_mean(y, axis=[3, 5], keepdims=True)
        y = tf.tile(y, [1, 1, 1, 2, 1, 2])
        y = tf.reshape(y, [-1, s[1], s[2], s[3]])
        x = tflib.lerp(x, y, lod - tf.floor(lod))
    with tf.name_scope('UpscaleLOD'): # Upscale to match the expected input/output size of the networks.
        s = tf.shape(x)
        factor = tf.cast(2 ** tf.floor(lod), tf.int32)
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x, labels

#----------------------------------------------------------------------------
# Evaluate time-varying training parameters.

def training_schedule(
    cur_nimg,
    training_set,
    lod_initial_resolution  = None,     # Image resolution used at the beginning.
    lod_training_kimg       = 600,      # Thousands of real images to show before doubling the resolution.
    lod_transition_kimg     = 600,      # Thousands of real images to show when fading in new layers.
    minibatch_size_base     = 32,       # Global minibatch size.
    minibatch_size_dict     = {},       # Resolution-specific overrides.
    minibatch_gpu_base      = 4,        # Number of samples processed at a time by one GPU.
    minibatch_gpu_dict      = {},       # Resolution-specific overrides.
    G_lrate_base            = 0.002,    # Learning rate for the generator.
    G_lrate_dict            = {},       # Resolution-specific overrides.
    D_lrate_base            = 0.002,    # Learning rate for the discriminator.
    D_lrate_dict            = {},       # Resolution-specific overrides.
    lrate_rampup_kimg       = 0,        # Duration of learning rate ramp-up.
    tick_kimg_base          = 4,        # Default interval of progress snapshots.
    tick_kimg_dict          = {8:28, 16:24, 32:20, 64:16, 128:12, 256:8, 512:6, 1024:4}): # Resolution-specific overrides.

    # Initialize result dict.
    s = dnnlib.EasyDict()
    s.kimg = cur_nimg / 1000.0

    # Training phase.
    phase_dur = lod_training_kimg + lod_transition_kimg
    phase_idx = int(np.floor(s.kimg / phase_dur)) if phase_dur > 0 else 0
    phase_kimg = s.kimg - phase_idx * phase_dur

    # Level-of-detail and resolution.
    if lod_initial_resolution is None:
        s.lod = 0.0
    else:
        s.lod = training_set.resolution_log2
        s.lod -= np.floor(np.log2(lod_initial_resolution))
        s.lod -= phase_idx
        if lod_transition_kimg > 0:
            s.lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
        s.lod = max(s.lod, 0.0)
    s.resolution = 2 ** (training_set.resolution_log2 - int(np.floor(s.lod)))

    # Minibatch size.
    s.minibatch_size = minibatch_size_dict.get(s.resolution, minibatch_size_base)
    s.minibatch_gpu = minibatch_gpu_dict.get(s.resolution, minibatch_gpu_base)

    # Learning rate.
    s.G_lrate = G_lrate_dict.get(s.resolution, G_lrate_base)
    s.D_lrate = D_lrate_dict.get(s.resolution, D_lrate_base)
    if lrate_rampup_kimg > 0:
        rampup = min(s.kimg / lrate_rampup_kimg, 1.0)
        s.G_lrate *= rampup
        s.D_lrate *= rampup

    # Other parameters.
    s.tick_kimg = tick_kimg_dict.get(s.resolution, tick_kimg_base)
    return s

def get_input_fn(training_set, batch_size):
    zz = training_set.get_minibatch_np(batch_size)
    features = zz[0]
    labels = zz[1]
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    def input_fn(params):
        batch_size = params["batch_size"]
        #features = {"x": np.ones((8, 3), dtype=np.float32)}
        #labels = np.ones((8, 1), dtype=np.float32)
        return dataset.repeat().batch(batch_size, drop_remainder=True)
    return dataset, input_fn

def get_input_fn(training_set):
    def input_fn(params):
        import pdb; pdb.set_trace()
        batch_size = params["batch_size"]
        features, labels = training_set.get_minibatch_np(batch_size)
        features = tf.cast(features, dtype=tf.float32)
        labels = tf.cast(labels, dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        return dataset.repeat().batch(batch_size, drop_remainder=True)
    return input_fn


def set_shapes(batch_size, num_channels, resolution, label_size, images, labels, transpose_input=False):
    """Statically set the batch_size dimension."""
    #import pdb; pdb.set_trace()
    if transpose_input:
        #if FLAGS.train_batch_size // FLAGS.num_cores > 8:
        if True:
            shape = [resolution, resolution, num_channels, batch_size]
        else:
            shape = [resolution, resolution, batch_size, num_channels]
        images.set_shape(images.get_shape().merge_with(tf.TensorShape(shape)))
        images = tf.reshape(images, [-1])
        labels.set_shape(labels.get_shape().merge_with(
            tf.TensorShape([batch_size, label_size])))
    else:
        images.set_shape(images.get_shape().merge_with(
            tf.TensorShape([batch_size, num_channels, resolution, resolution])))
        labels.set_shape(labels.get_shape().merge_with(
            tf.TensorShape([batch_size, label_size])))
    return images, labels

import functools
from tensorflow.python.platform import gfile

def get_input_fn(load_training_set, num_cores, mirror_augment, drange_net):
    self = training_set = load_training_set(batch_size=0)

    def input_fn(params):
        batch_size = params["batch_size"]
        set_num_replicas(params["context"].num_replicas if "context" in params else 1)
        num_channels = training_set.shape[0]
        resolution = training_set.shape[1]
        resolution_log2 = int(np.log2(resolution))
        label_size = training_set.label_size
        assert gfile.IsDirectory(training_set.tfrecord_dir)
        tfr_files = sorted(tf.io.gfile.glob(os.path.join(training_set.tfrecord_dir, '*-r%02d.tfrecords' % resolution_log2)))
        assert len(tfr_files) == 1

        lod_index = -1
        for tfr_file, tfr_shape, tfr_lod in training_set.tfr:
          if tfr_shape[1] == resolution and tfr_shape[2] == resolution:
            lod_index = tfr_lod
        assert lod_index >= 0

        dset = tf.data.Dataset.from_tensor_slices(tfr_files)
        dset = dset.apply(tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=4, sloppy=True))

        if training_set.label_file is not None:
            training_set._tf_labels_var, training_set._tf_labels_init = tflib.create_var_with_large_initial_value2(training_set._np_labels, name='labels_var', trainable=False)
            with tf.control_dependencies([training_set._tf_labels_init]):
                training_set._tf_labels_dataset = tf.data.Dataset.from_tensor_slices(training_set._tf_labels_var)
        else:
            training_set._tf_labels_dataset = tf.data.Dataset.from_tensor_slices(training_set._np_labels)

        dset = dset.map(dataset.TFRecordDataset.parse_tfrecord_tf, num_parallel_calls=2)
        dset = tf.data.Dataset.zip((dset, training_set._tf_labels_dataset))

        shuffle_mb = 4096  # Shuffle data within specified window (megabytes), 0 = disable shuffling.
        prefetch_mb = 2048  # Amount of data to prefetch (megabytes), 0 = disable prefetching.

        tfr_shape = (resolution, resolution, 3)
        bytes_per_item = np.prod(tfr_shape) * np.dtype(training_set.dtype).itemsize
        if shuffle_mb > 0:
            dset = dset.shuffle(((shuffle_mb << 20) - 1) // bytes_per_item + 1)
        repeat = True
        if repeat:
            dset = dset.repeat()
        if prefetch_mb > 0:
            dset = dset.prefetch(((prefetch_mb << 20) - 1) // bytes_per_item + 1)
        dset = dset.batch(batch_size)

        def dataset_parser_dynamic(features, labels):
            features, labels = process_reals(features, labels, lod=0.0, mirror_augment=mirror_augment, drange_data=training_set.dynamic_range, drange_net=drange_net)
            return features, labels

        dset = dset.map(
                dataset_parser_dynamic,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Assign static batch size dimension
        dset = dset.map(functools.partial(set_shapes, batch_size, num_channels, resolution, label_size))

        return dset
    return input_fn, training_set

#----------------------------------------------------------------------------
# Main training script.

def training_loop(
    G_args                  = {},       # Options for generator network.
    D_args                  = {},       # Options for discriminator network.
    G_opt_args              = {},       # Options for generator optimizer.
    D_opt_args              = {},       # Options for discriminator optimizer.
    G_loss_args             = {},       # Options for generator loss.
    D_loss_args             = {},       # Options for discriminator loss.
    dataset_args            = {},       # Options for dataset.load_dataset().
    sched_args              = {},       # Options for train.TrainingSchedule.
    grid_args               = {},       # Options for train.setup_snapshot_image_grid().
    metric_arg_list         = [],       # Options for MetricGroup.
    tf_config               = {},       # Options for tflib.init_tf().
    data_dir                = None,     # Directory to load datasets from.
    G_smoothing_kimg        = 10.0,     # Half-life of the running average of generator weights.
    minibatch_repeats       = 4,        # Number of minibatches to run before adjusting training parameters.
    lazy_regularization     = False,    # Perform regularization as a separate training step?
    G_reg_interval          = 4,        # How often the perform regularization for G? Ignored if lazy_regularization=False.
    D_reg_interval          = 16,       # How often the perform regularization for D? Ignored if lazy_regularization=False.
    reset_opt_for_new_lod   = True,     # Reset optimizer internal state (e.g. Adam moments) when new layers are introduced?
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    mirror_augment          = False,    # Enable mirror augment?
    drange_net              = [-1,1],   # Dynamic range used when feeding image data to the networks.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = only save 'reals.png' and 'fakes-init.png'.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = only save 'networks-final.pkl'.
    save_tf_graph           = False,    # Include full TensorFlow computation graph in the tfevents file?
    save_weight_histograms  = False,    # Include weight histograms in the tfevents file?
    resume_pkl              = None,     # Network pickle to resume training from, None = train from scratch.
    resume_kimg             = 0.0,      # Assumed training progress at the beginning. Affects reporting and training schedule.
    resume_time             = 0.0,      # Assumed wallclock time at the beginning. Affects reporting.
    resume_with_new_nets    = False):   # Construct new networks according to G_args and D_args before resuming training?

    tf.logging.set_verbosity(tf.logging.INFO)

    if resume_pkl is None and 'RESUME_PKL' in os.environ:
        resume_pkl = os.environ['RESUME_PKL']
    if resume_kimg <= 0.0 and 'RESUME_KIMG' in os.environ:
        resume_kimg = float(os.environ['RESUME_KIMG'])
    if resume_time <= 0.0 and 'RESUME_TIME' in os.environ:
        resume_time = float(os.environ['RESUME_TIME'])

    # Initialize dnnlib and TensorFlow.
    #tflib.init_tf(tf_config)
    num_gpus = dnnlib.submit_config.num_gpus

    # Load training set.
    #training_set = dataset.load_dataset(data_dir=dnnlib.convert_path(data_dir), verbose=True, **dataset_args)
    #import pdb; pdb.set_trace()
    #dset, input_fn = get_input_fn(training_set, num_gpus*32)
    def load_training_set(**kws):
        return dataset.load_dataset(data_dir=dnnlib.convert_path(data_dir), verbose=True, **dataset_args, **kws)
    input_fn, training_set = get_input_fn(load_training_set, num_gpus, mirror_augment=mirror_augment, drange_net=drange_net)

    def model_fn(features, labels, mode, params):
        nonlocal G_opt_args, D_opt_args, sched_args, G_reg_interval, D_reg_interval, lazy_regularization, G_smoothing_kimg
        assert mode == tf.estimator.ModeKeys.TRAIN
        num_channels = features.shape[1].value
        resolution = features.shape[2].value
        label_size = labels.shape[1].value
        G = tflib.Network('G', num_channels=num_channels, resolution=resolution, label_size=label_size, **G_args)
        D = tflib.Network('D', num_channels=num_channels, resolution=resolution, label_size=label_size, **D_args)
        G.print_layers(); D.print_layers()
        Gs, Gs_finalize = G.clone2('Gs')
        G_gpu = G
        D_gpu = D
        reals_read = features
        labels_read = labels
        minibatch_gpu_in = params['batch_size']
        G_opt_args = dict(G_opt_args)
        D_opt_args = dict(D_opt_args)
        sched_args = dict(sched_args)
        G_opt = tflib.Optimizer(name='TrainG', cross_shard=True, learning_rate=sched_args['G_lrate_base'], **G_opt_args)
        D_opt = tflib.Optimizer(name='TrainD', cross_shard=True, learning_rate=sched_args['D_lrate_base'], **D_opt_args)
        with tf.name_scope('G_loss'):
            G_loss, G_reg = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, opt=G_opt, training_set=training_set,
                                                          minibatch_size=minibatch_gpu_in, **G_loss_args)
        with tf.name_scope('D_loss'):
            D_loss, D_reg = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, opt=D_opt, training_set=training_set,
                                                          minibatch_size=minibatch_gpu_in, reals=reals_read,
                                                          labels=labels_read, **D_loss_args)

        # Register gradients.
        #G_reg_interval = tf.constant(G_reg_interval, tf.int64)
        #D_reg_interval = tf.constant(D_reg_interval, tf.int64)
        #zero = tf.constant(0, tf.int64)
        #one = tf.constant(1, tf.int64)
        G_reg_opt = tflib.Optimizer(name='RegG', share=G_opt, cross_shard=True, learning_rate=sched_args['G_lrate_base'], **G_opt_args)
        D_reg_opt = tflib.Optimizer(name='RegD', share=D_opt, cross_shard=True, learning_rate=sched_args['D_lrate_base'], **D_opt_args)
        if not lazy_regularization:
            G_reg_loss = None
            D_reg_loss = None
            if G_reg is not None: G_loss += G_reg
            if D_reg is not None: D_loss += D_reg
        else:
            if G_reg is not None: G_reg_loss = tf.reduce_mean(G_reg * G_reg_interval)
            if D_reg is not None: D_reg_loss = tf.reduce_mean(D_reg * D_reg_interval)
            if G_reg is not None: G_reg_opt.register_gradients(G_reg_loss, G_gpu.trainables)
            if D_reg is not None: D_reg_opt.register_gradients(D_reg_loss, D_gpu.trainables)
        G_loss_op = tf.reduce_mean(G_loss)
        D_loss_op = tf.reduce_mean(D_loss)
        G_opt.register_gradients(G_loss_op, G_gpu.trainables)
        D_opt.register_gradients(D_loss_op, D_gpu.trainables)
        inc_global_step = tf.assign_add(tf.train.get_or_create_global_step(), minibatch_gpu_in, name="inc_global_step")
        G_train_op = G_opt._shared_optimizers[''].minimize(G_loss_op, var_list=G_gpu.trainables)
        D_train_op = D_opt._shared_optimizers[''].minimize(D_loss_op, var_list=D_gpu.trainables)
        if lazy_regularization:
            assert (G_reg is not None)
            assert (D_reg is not None)
            def G_fn():
                with tf.control_dependencies([tf.train.get_or_create_global_step()]):
                    G_op = G_reg_opt._shared_optimizers[''].minimize(G_reg_loss, var_list=G_gpu.trainables)
                    return G_op
            def D_fn():
                with tf.control_dependencies([tf.train.get_or_create_global_step()]):
                    D_op = D_reg_opt._shared_optimizers[''].minimize(D_reg_loss, var_list=D_gpu.trainables)
                    return D_op
            def G_else():
                with tf.control_dependencies([tf.train.get_or_create_global_step()]):
                    return tf.no_op()
            def D_else():
                with tf.control_dependencies([tf.train.get_or_create_global_step()]):
                    return tf.no_op()
            #G_reg_train_op = tf.cond(lambda: tf.equal(tf.cast(tf.mod(tf.cast(tf.train.get_or_create_global_step(), tf.int32), tf.cast(G_reg_interval, tf.int32)), tf.int32), tf.constant(0, tf.int32)), G_fn, G_else)
            #D_reg_train_op = tf.cond(lambda: tf.equal(tf.cast(tf.mod(tf.cast(tf.train.get_or_create_global_step(), tf.int32), tf.cast(D_reg_interval, tf.int32)), tf.int32), tf.constant(0, tf.int32)), D_fn, D_else)
            G_reg_train_op = tf.cond(lambda: tf.equal(tf.mod(tf.train.get_or_create_global_step(), G_reg_interval), tf.constant(0, tf.int64)), G_fn, G_else)
            D_reg_train_op = tf.cond(lambda: tf.equal(tf.mod(tf.train.get_or_create_global_step(), D_reg_interval), tf.constant(0, tf.int64)), D_fn, D_else)
        else:
            G_reg_train_op = G_reg_opt._shared_optimizers[''].minimize(G_reg_loss, var_list=G_gpu.trainables) if G_reg_loss is not None else tf.no_op()
            D_reg_train_op = D_reg_opt._shared_optimizers[''].minimize(D_reg_loss, var_list=D_gpu.trainables) if D_reg_loss is not None else tf.no_op()

        #run_G_reg = (lazy_regularization and running_mb_counter % G_reg_interval == 0)
        #run_D_reg = (lazy_regularization and running_mb_counter % D_reg_interval == 0)

        minibatch_size_in = batch_size
        Gs_beta              = 0.5 ** tf.div(tf.cast(minibatch_size_in, tf.float32), G_smoothing_kimg * 1000.0) if G_smoothing_kimg > 0.0 else 0.0
        Gs_update_op = Gs.setup_as_moving_average_of(G, beta=Gs_beta)
        loss = G_loss_op + D_loss_op
        if G_reg_loss is not None: loss += G_reg_loss
        if D_reg_loss is not None: loss += D_reg_loss
        with tf.control_dependencies([inc_global_step]):
            with tf.control_dependencies([G_train_op]):
                with tf.control_dependencies([G_reg_train_op]):
                    with tf.control_dependencies([D_train_op]):
                        with tf.control_dependencies([D_reg_train_op]):
                            train_op = tf.group(Gs_update_op, name='train_op')
        #import pdb; pdb.set_trace()

        #train_op = G_opt.minimize(G_loss, global_step=tf.train.get_or_create_global_step())
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)

        #import pdb; pdb.set_trace()

    use_tpu = True
    training_steps = 2048*20480
    batch_size = sched_args.minibatch_size_base
    pprint(sched_args)
    model_dir=os.environ['MODEL_DIR'] if 'MODEL_DIR' in os.environ else 'gs://danbooru-euw4a/test/run30/'
    #if tf.gfile.Exists(model_dir):
    #  tf.gfile.DeleteRecursively(model_dir)
    tpu_cluster_resolver = tflex.get_tpu_resolver()
    run_config = tf.contrib.tpu.RunConfig(
        model_dir=model_dir,
        #save_checkpoints_steps=100,
        save_checkpoints_secs=600//5,
        keep_checkpoint_max=10,
        keep_checkpoint_every_n_hours=2,
        cluster=tpu_cluster_resolver,
        tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=256))
    estimator = tf.contrib.tpu.TPUEstimator(
        config=run_config,
        use_tpu=use_tpu,
        model_fn=model_fn,
        train_batch_size=batch_size)
    #import pdb; pdb.set_trace()
    print('Training...')
    estimator.train(input_fn, steps=training_steps)
    import pdb; pdb.set_trace()

    grid_size, grid_reals, grid_labels = misc.setup_snapshot_image_grid(training_set, **grid_args)
    misc.save_image_grid(grid_reals, dnnlib.make_run_dir_path('reals.png'), drange=training_set.dynamic_range, grid_size=grid_size)

    # Construct or load networks.
    with tflex.device('/gpu:0'):
        if resume_pkl is None or resume_with_new_nets:
            print('Constructing networks...')
            G = tflib.Network('G', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **G_args)
            D = tflib.Network('D', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **D_args)
            Gs = G.clone('Gs')
        if resume_pkl is not None:
            print('Loading networks from "%s"...' % resume_pkl)
            rG, rD, rGs = misc.load_pkl(resume_pkl)
            if resume_with_new_nets: G.copy_vars_from(rG); D.copy_vars_from(rD); Gs.copy_vars_from(rGs)
            else: G = rG; D = rD; Gs = rGs

    # Print layers and generate initial image snapshot.
    G.print_layers(); D.print_layers()
    sched = training_schedule(cur_nimg=total_kimg*1000, training_set=training_set, **sched_args)
    grid_latents = np.random.randn(np.prod(grid_size), *G.input_shape[1:])

    def save_image_grid(latents, grid_size, filename):
        grid_fakes = Gs.run(latents, grid_labels, is_validation=True, randomize_noise=False, minibatch_size=sched.minibatch_gpu)
        misc.save_image_grid(grid_fakes, filename, drange=drange_net, grid_size=grid_size)

    tflex.save_image_grid = save_image_grid

    def save_image_grid_command(randomize=False):
        if randomize:
            tflex.latents = np.random.randn(np.prod(grid_size), *G.input_shape[1:])
        if not hasattr(tflex, 'latents'):
            tflex.latents = grid_latents
        if randomize or not hasattr(tflex, 'grid_filename'):
            tflex.grid_filename = dnnlib.make_run_dir_path('grid%06d.png' % int(time.time()))
        use_grid_size = (2, 2)
        latents = tflex.latents[:np.prod(use_grid_size)]
        tflex.save_image_grid(latents, use_grid_size, tflex.grid_filename)
        print('Saved ' + tflex.grid_filename)

    tflex.save_image_grid_command = save_image_grid_command

    @tflex.register_command
    def image_grid():
        tflex.save_image_grid_command(randomize=True)

    @tflex.register_command
    def resave_image_grid():
        tflex.save_image_grid_command(randomize=False)

    # Setup training inputs.
    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'), tflex.device('/cpu:0'):
        lod_in               = tf.placeholder(tf.float32, name='lod_in', shape=[])
        lrate_in             = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_size_in    = tf.placeholder(tf.int32, name='minibatch_size_in', shape=[])
        minibatch_gpu_in     = tf.placeholder(tf.int32, name='minibatch_gpu_in', shape=[])
        minibatch_multiplier = minibatch_size_in // (minibatch_gpu_in * num_gpus)
        Gs_beta              = 0.5 ** tf.div(tf.cast(minibatch_size_in, tf.float32), G_smoothing_kimg * 1000.0) if G_smoothing_kimg > 0.0 else 0.0

    # Setup optimizers.
    G_opt_args = dict(G_opt_args)
    D_opt_args = dict(D_opt_args)
    for args, reg_interval in [(G_opt_args, G_reg_interval), (D_opt_args, D_reg_interval)]:
        args['minibatch_multiplier'] = minibatch_multiplier
        args['learning_rate'] = lrate_in
        if lazy_regularization:
            mb_ratio = reg_interval / (reg_interval + 1)
            args['learning_rate'] *= mb_ratio
            if 'beta1' in args: args['beta1'] **= mb_ratio
            if 'beta2' in args: args['beta2'] **= mb_ratio
    G_opt = tflib.Optimizer(name='TrainG', **G_opt_args)
    D_opt = tflib.Optimizer(name='TrainD', **D_opt_args)
    G_reg_opt = tflib.Optimizer(name='RegG', share=G_opt, **G_opt_args)
    D_reg_opt = tflib.Optimizer(name='RegD', share=D_opt, **D_opt_args)

    # Build training graph for each GPU.
    data_fetch_ops = []
    shards = {}
    def get_shard(i):
        with tflex.lock:
            if i not in shards:
                shards[i] = dnnlib.EasyDict()
            return shards[i]
    def make_generator(gpu):
        me = get_shard(gpu)
        prev = None if gpu <= 0 else get_shard(gpu - 1)
        with tf.name_scope('GPU%d' % gpu), tflex.device('/gpu:%d' % gpu):
            # Create GPU-specific shadow copies of G and D.
            G_gpu, G_final = (G, None) if gpu == 0 else G.clone2(G.name + '_shadow')
            D_gpu, D_final = (D, None) if gpu == 0 else D.clone2(D.name + '_shadow')
        with tflex.lock:
            me.G = G_gpu
            me.D = D_gpu
            me.G_final = tflex.defer(G_final, dependencies=[prev.G_final] if prev else [])
            me.D_final = tflex.defer(D_final, dependencies=[prev.D_final] if prev else [])
    print('Making generators...')
    tflex.parallelize_verbose("Generator", range(num_gpus), make_generator, synchronous=True)

    def make_shard(gpu):
        nonlocal data_fetch_ops
        me = get_shard(gpu)
        with tf.name_scope('GPU%d' % gpu), tflex.device('/gpu:%d' % gpu):
            # Create GPU-specific shadow copies of G and D.
            G_gpu = me.G
            D_gpu = me.D
            me.G_final.join()
            me.D_final.join()

            tflex.break_next_run = (gpu > 0)

            # Fetch training data via temporary variables.
            with tf.name_scope('DataFetch'):
                sched = training_schedule(cur_nimg=int(resume_kimg*1000), training_set=training_set, **sched_args)
                reals_var = tf.Variable(name='reals', trainable=False, initial_value=tf.zeros([sched.minibatch_gpu] + training_set.shape))
                labels_var = tf.Variable(name='labels', trainable=False, initial_value=tf.zeros([sched.minibatch_gpu, training_set.label_size]))
                reals_write, labels_write = training_set.get_minibatch_tf()
                reals_write, labels_write = process_reals(reals_write, labels_write, lod_in, mirror_augment, training_set.dynamic_range, drange_net)
                reals_write = tf.concat([reals_write, reals_var[minibatch_gpu_in:]], axis=0)
                labels_write = tf.concat([labels_write, labels_var[minibatch_gpu_in:]], axis=0)
                data_fetch_ops += [tf.group(tf.assign(reals_var, reals_write), name="fetch_reals")]
                data_fetch_ops += [tf.group(tf.assign(labels_var, labels_write), name="fetch_labels")]
                reals_read = reals_var[:minibatch_gpu_in]
                labels_read = labels_var[:minibatch_gpu_in]

            # Evaluate loss functions.
            lod_assign_ops = []
            if 'lod' in G_gpu.vars: lod_assign_ops += [tf.assign(G_gpu.vars['lod'], lod_in)]
            if 'lod' in D_gpu.vars: lod_assign_ops += [tf.assign(D_gpu.vars['lod'], lod_in)]
            with tf.control_dependencies(lod_assign_ops):
                with tf.name_scope('G_loss'):
                    G_loss, G_reg = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, opt=G_opt, training_set=training_set, minibatch_size=minibatch_gpu_in, **G_loss_args)
                with tf.name_scope('D_loss'):
                    D_loss, D_reg = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, opt=D_opt, training_set=training_set, minibatch_size=minibatch_gpu_in, reals=reals_read, labels=labels_read, **D_loss_args)

            # Register gradients.
            if not lazy_regularization:
                if G_reg is not None: G_loss += G_reg
                if D_reg is not None: D_loss += D_reg
            else:
                if G_reg is not None: G_reg_opt.register_gradients(tf.reduce_mean(G_reg * G_reg_interval), G_gpu.trainables)
                if D_reg is not None: D_reg_opt.register_gradients(tf.reduce_mean(D_reg * D_reg_interval), D_gpu.trainables)
            G_opt.register_gradients(tf.reduce_mean(G_loss), G_gpu.trainables)
            D_opt.register_gradients(tf.reduce_mean(D_loss), D_gpu.trainables)
    print('Making shards...')
    tflex.parallelize_verbose("Shard", range(num_gpus), make_shard, synchronous=True)

    # Setup training ops.
    data_fetch_op = tf.group(*data_fetch_ops)
    G_train_op, G_finalize = G_opt.apply_updates()
    D_train_op, D_finalize = D_opt.apply_updates()
    G_reg_op, G_reg_finalize = G_reg_opt.apply_updates(allow_no_op=True)
    D_reg_op, D_reg_finalize = D_reg_opt.apply_updates(allow_no_op=True)
    Gs_update_op = Gs.setup_as_moving_average_of(G, beta=Gs_beta)

    tflex.parallelize_verbose("initialize optimizers", [G_finalize, D_finalize, G_reg_finalize, D_reg_finalize], lambda f: f())

    # Finalize graph.
    if tflex.has_gpu():
        with tflex.device('/gpu:0'):
            try:
                peak_gpu_mem_op = tf.contrib.memory_stats.MaxBytesInUse()
            except tf.errors.NotFoundError:
                peak_gpu_mem_op = tf.constant(0)
    else:
        peak_gpu_mem_op = None
    tflib.init_uninitialized_vars()

    print('Initializing logs...')
    summary_log = tf.summary.FileWriter(dnnlib.make_run_dir_path())
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        G.setup_weight_histograms(); D.setup_weight_histograms()
    metrics = metric_base.MetricGroup(metric_arg_list)

    print('Training for %d kimg...\n' % total_kimg)
    dnnlib.RunContext.get().update('', cur_epoch=resume_kimg, max_epoch=total_kimg)
    maintenance_time = dnnlib.RunContext.get().get_last_update_interval()
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = -1
    tick_start_nimg = cur_nimg
    prev_lod = -1.0
    running_mb_counter = 0
    need_warmup = True
    while cur_nimg < total_kimg * 1000:
        if tflex.state.noisy: print('cur_nimg', cur_nimg, total_kimg)
        if dnnlib.RunContext.get().should_stop(): break

        # Choose training parameters and configure training ops.
        sched = training_schedule(cur_nimg=cur_nimg, training_set=training_set, **sched_args)
        assert sched.minibatch_size % (sched.minibatch_gpu * num_gpus) == 0
        training_set.configure(sched.minibatch_gpu, sched.lod)
        if reset_opt_for_new_lod:
            if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(sched.lod) != np.ceil(prev_lod):
                G_opt.reset_optimizer_state(); D_opt.reset_optimizer_state()
        prev_lod = sched.lod

        # Run training ops.
        feed_dict = {lod_in: sched.lod, lrate_in: sched.G_lrate, minibatch_size_in: sched.minibatch_size, minibatch_gpu_in: sched.minibatch_gpu}
        if need_warmup:
            i = 0
            all_ops = [[G_train_op, data_fetch_op], [G_reg_op], [D_train_op, Gs_update_op], [D_reg_op]]
            for ops in all_ops:
                tflex.parallelize_verbose("warmup %d / %d" % (i, len(all_ops)), ops, lambda op: tflib.run(op, feed_dict))
                i += 1
            need_warmup = False
        for _repeat in range(minibatch_repeats):
            if tflex.state.noisy: print('_repeat', _repeat)
            rounds = range(0, sched.minibatch_size, sched.minibatch_gpu * num_gpus)
            run_G_reg = (lazy_regularization and running_mb_counter % G_reg_interval == 0)
            run_D_reg = (lazy_regularization and running_mb_counter % D_reg_interval == 0)
            cur_nimg += sched.minibatch_size
            running_mb_counter += 1

            # Fast path without gradient accumulation.
            if len(rounds) == 1:
                if tflex.state.noisy: print('G_train_op', 'fast path')
                tflib.run(G_train_op, feed_dict)
                tflib.run(data_fetch_op, feed_dict)
                if run_G_reg:
                    tflib.run(G_reg_op, feed_dict)
                if tflex.state.noisy: print('D_train_op', 'fast path')
                tflib.run(D_train_op, feed_dict)
                tflib.run(Gs_update_op, feed_dict)
                if run_D_reg:
                    tflib.run(D_reg_op, feed_dict)

            # Slow path with gradient accumulation.
            else:
                for _round in rounds:
                    if tflex.state.noisy: print('G_train_op', 'slow path')
                    tflib.run(G_train_op, feed_dict)
                if run_G_reg:
                    for _round in rounds:
                        if tflex.state.noisy: print('G_reg_op', 'slow path')
                        tflib.run(G_reg_op, feed_dict)
                if tflex.state.noisy: print('G_update_op', 'slow path')
                tflib.run(Gs_update_op, feed_dict)
                for _round in rounds:
                    if tflex.state.noisy: print('data_fetch_op', 'slow path')
                    tflib.run(data_fetch_op, feed_dict)
                    if tflex.state.noisy: print('D_train_op', 'slow path')
                    tflib.run(D_train_op, feed_dict)
                if run_D_reg:
                    for _round in rounds:
                        if tflex.state.noisy: print('D_reg_op', 'slow path')
                        tflib.run(D_reg_op, feed_dict)

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_tick < 0 or cur_nimg >= tick_start_nimg + sched.tick_kimg * 1000 or done:
            cur_tick += 1
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_time = dnnlib.RunContext.get().get_time_since_last_update()

            def report_progress_command():
                total_time = dnnlib.RunContext.get().get_time_since_start() + resume_time
                tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
                tick_time = dnnlib.RunContext.get().get_time_since_last_update()
                print('tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %-6.1f gpumem %.1f' % (
                    autosummary('Progress/tick', cur_tick),
                    autosummary('Progress/kimg', cur_nimg / 1000.0),
                    autosummary('Progress/lod', sched.lod),
                    autosummary('Progress/minibatch', sched.minibatch_size),
                    dnnlib.util.format_time(autosummary('Timing/total_sec', total_time)),
                    autosummary('Timing/sec_per_tick', tick_time),
                    autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                    autosummary('Timing/maintenance_sec', maintenance_time),
                    autosummary('Resources/peak_gpu_mem_gb', (peak_gpu_mem_op.eval() if peak_gpu_mem_op is not None else 0) / 2**30)))
                autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
                autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))

            if not hasattr(tflex, 'report_progress_command'):
                tflex.report_progress_command = report_progress_command

            @tflex.register_command
            def report_progress():
                tflex.report_progress_command()

            def save_command():
                pkl = dnnlib.make_run_dir_path('network-snapshot-%06d.pkl' % (cur_nimg // 1000))
                misc.save_pkl((G, D, Gs), pkl)
                metrics.run(pkl, run_dir=dnnlib.make_run_dir_path(), data_dir=dnnlib.convert_path(data_dir), num_gpus=num_gpus, tf_config=tf_config)

            if not hasattr(tflex, 'save_command'):
                tflex.save_command = save_command

            @tflex.register_command
            def save():
                tflex.save_command()

            try:
              # Report progress.
              tflex.report_progress_command()
              tick_start_nimg = cur_nimg

              # Save snapshots.
              if image_snapshot_ticks is not None and (cur_tick % image_snapshot_ticks == 0 or done):
                  def thunk(_):
                      grid_fakes = Gs.run(grid_latents, grid_labels, is_validation=True, randomize_noise=False, minibatch_size=sched.minibatch_gpu)
                      misc.save_image_grid(grid_fakes, dnnlib.make_run_dir_path('fakes%06d.png' % (cur_nimg // 1000)), drange=drange_net, grid_size=grid_size)
                  tflex.parallelize([0], thunk)
              if network_snapshot_ticks is not None and cur_tick > 0 and (cur_tick % network_snapshot_ticks == 0 or done):
                  def thunk(_):
                      tflex.save_command()
                  tflex.parallelize([0], thunk)

              # Update summaries and RunContext.
              metrics.update_autosummaries()
              tflib.autosummary.save_summaries(summary_log, cur_nimg)
              dnnlib.RunContext.get().update('%.2f' % sched.lod, cur_epoch=cur_nimg // 1000, max_epoch=total_kimg)
              maintenance_time = dnnlib.RunContext.get().get_last_update_interval() - tick_time
            except:
              traceback.print_exc()

    # Save final snapshot.
    misc.save_pkl((G, D, Gs), dnnlib.make_run_dir_path('network-final.pkl'))

    # All done.
    summary_log.close()
    training_set.close()

#----------------------------------------------------------------------------
