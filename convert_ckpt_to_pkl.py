import argparse
import pickle
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import tensorflow as tf
from training import misc

def main():
    parser = argparse.ArgumentParser(
        description='Creates a pkl from a ckpt of a StyleGAN2 model using a reference pkl of a network of the same size.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--ckpt_model_dir', help='The directory with the ckpt files', required=True)
    parser.add_argument('--reference_pkl', help='A reference pkl of a StyleGAN2, must have the exact same variables as ckpt (will not be overwritten)', required=True)

    args = parser.parse_args()

    model_dir = args.ckpt_model_dir
    name = args.reference_pkl

    tflib.init_tf()
    G, D, Gs = pickle.load(open(name, "rb"))
    G.print_layers(); D.print_layers(); Gs.print_layers()

    var_list = [v for v in tf.global_variables()]
    saver = tf.train.Saver(
      var_list=var_list,
    )
    ckpt = tf.train.latest_checkpoint(model_dir)
    sess = tf.get_default_session()
    saver.restore(sess, ckpt)

    out_pkl_iteration = ckpt.split('ckpt-')[-1]
    out_pkl = './model.ckpt-'+out_pkl_iteration+'.pkl'
    print('Saving %s' % out_pkl)
    misc.save_pkl((G, D, Gs), out_pkl)


if __name__ == '__main__':
    main()