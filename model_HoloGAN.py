from __future__ import division
from tools.model_utils import transform_voxel_to_match_image
from tools.rotation_utils import *
from tools.utils import get_image, merge, inverse_transform, to_bool
from tools.ops import *
import os
import sys
import glob
import json
import shutil
import imageio
import numpy as np

with open(sys.argv[1], 'r') as fh:
    cfg = json.load(fh)
IMAGE_PATH = cfg['image_path']
OUTPUT_DIR = cfg['output_dir']
Z_MAP_IMAGE_PATH = cfg['z_map_image_path']

LOGDIR = os.path.join(OUTPUT_DIR, "log")
MODELDIR = os.path.join(OUTPUT_DIR, 'models')
IMG_DIR = os.path.join(OUTPUT_DIR, 'images')
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples")

# ----------------------------------------------------------------------------

class HoloGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, crop=True,
                 output_height=64, output_width=64,
                 gf_dim=64, df_dim=64,
                 c_dim=3, dataset_name='no dataset name',
                 input_fname_pattern='*.webp'):

        self.sess = sess
        self.crop = crop

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.data = glob.glob(os.path.join(
            IMAGE_PATH, self.input_fname_pattern))
        self.z_map_image = glob.glob(os.path.join(
            Z_MAP_IMAGE_PATH, self.input_fname_pattern))
        self.model_dir = os.path.join(MODELDIR, self.dataset_name)
        self.img_dir = os.path.join(IMG_DIR, self.dataset_name)
        self.log_dir = os.path.join(LOGDIR, self.dataset_name)
        self.sample_dir = os.path.join(SAMPLE_DIR, self.dataset_name)
        if not os.path.exists(self.model_dir):
          os.makedirs(self.model_dir)
        if not os.path.exists(self.img_dir):
          os.makedirs(self.img_dir)
        if not os.path.exists(self.log_dir):
          os.makedirs(self.log_dir)


    def build(self, build_func_name):
        build_func = eval("self." + build_func_name)
        build_func()

    def build_HoloGAN(self):
        self.view_in = tf.compat.v1.placeholder(tf.float32, [None, 6], name='view_in')
        self.inputs = tf.compat.v1.placeholder(tf.float32, [
                                     None, self.output_height, self.output_width, self.c_dim], name='real_images')
        self.z = tf.compat.v1.placeholder(tf.float32, [None, cfg['z_dim']], name='z')
        #self.z_map = tf.Variable(tf.ones([1, cfg['z_dim']]))
        inputs = self.inputs

        gen_func = eval("self." + (cfg['generator']))
        dis_func = eval("self." + (cfg['discriminator']))
        self.gen_view_func = eval(cfg['view_func'])

        self.G = gen_func(self.z, self.view_in)

        if str.lower(str(cfg["style_disc"])) == "true":
            print("Style Disc")
            self.D, self.D_logits, _, self.d_h1_r, self.d_h2_r, self.d_h3_r, self.d_h4_r = dis_func(
                inputs, cont_dim=cfg['z_dim'], reuse=False)
            self.D_, self.D_logits_, self.Q_c_given_x, self.d_h1_f, self.d_h2_f, self.d_h3_f, self.d_h4_f = dis_func(
                self.G, cont_dim=cfg['z_dim'], reuse=True)

            self.d_h1_loss = cfg["DStyle_lambda"] * (
                tf.reduce_mean(sigmoid_cross_entropy_with_logits(
                    self.d_h1_r, tf.ones_like(self.d_h1_r)))
                + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_f, tf.zeros_like(self.d_h1_f))))
            self.d_h2_loss = cfg["DStyle_lambda"] * (
                tf.reduce_mean(sigmoid_cross_entropy_with_logits(
                    self.d_h2_r, tf.ones_like(self.d_h2_r)))
                + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h2_f, tf.zeros_like(self.d_h2_f))))
            self.d_h3_loss = cfg["DStyle_lambda"] * (
                tf.reduce_mean(sigmoid_cross_entropy_with_logits(
                    self.d_h3_r, tf.ones_like(self.d_h3_r)))
                + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h3_f, tf.zeros_like(self.d_h3_f))))
            self.d_h4_loss = cfg["DStyle_lambda"] * (
                tf.reduce_mean(sigmoid_cross_entropy_with_logits(
                    self.d_h4_r, tf.ones_like(self.d_h4_r)))
                + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h4_f, tf.zeros_like(self.d_h4_f))))
        else:
            self.D, self.D_logits, _ = dis_func(
                inputs, cont_dim=cfg['z_dim'], reuse=False)
            self.D_, self.D_logits_, self.Q_c_given_x = dis_func(
                self.G, cont_dim=cfg['z_dim'], reuse=True)

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(
            self.D_logits_, tf.ones_like(self.D_)))
        
        #self.G_z_map = gen_func(self.z_map, self.view_in, reuse=True)

        if str.lower(str(cfg["style_disc"])) == "true":
            print("Style disc")
            self.d_loss = self.d_loss + self.d_h1_loss + \
                self.d_h2_loss + self.d_h3_loss + self.d_h4_loss
        # ====================================================================================================================
        # Identity loss

        self.q_loss = cfg["lambda_latent"] * \
            tf.reduce_mean(tf.square(self.Q_c_given_x - self.z))
        self.d_loss = self.d_loss + self.q_loss
        self.g_loss = self.g_loss + self.q_loss

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()
 
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()        

    
    def train_z_map(self, config):
        sample_z = self.sampling_Z(cfg['z_dim'], str(cfg['sample_z']))
        #z_var = tf.Variable(sample_z, name="z_var")
        sample_view = self.gen_view_func(cfg['batch_size'],
                                                cfg['ele_low'], cfg['ele_high'],
                                                cfg['azi_low'], cfg['azi_high'],
                                                cfg['scale_low'], cfg['scale_high'],
                                                cfg['x_low'], cfg['x_high'],
                                                cfg['y_low'], cfg['y_high'],
                                                cfg['z_low'], cfg['z_high'],
                                                with_translation=False,
                                                with_scale=to_bool(str(cfg['with_translation'])))
        sample_file = self.z_map_image[0]
        sample_image = get_image(sample_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=False)
        sample_image = sample_image.reshape(1, 64,64,3).astype('float32')

        could_load, checkpoint_counter = self.load()
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return


        vars = tf.trainable_variables()
        z_var = [var for var in vars if 'z_weight' in var.name]
        print('z var: ', self.sess.run(z_var))
        raise Exception('hej')
        
        mae = tf.keras.losses.MeanAbsoluteError(reduction="sum")
        target_image_difference = mae(sample_image[0], self.G)
        regularizer = tf.abs(tf.norm(z_var) - np.sqrt(cfg['z_dim']))
        regularizer = tf.cast(regularizer, dtype=tf.float32)
        z_map_loss = target_image_difference + regularizer
        optimizer = tf.train.AdamOptimizer(learning_rate=0.009, name="z_map_optimizer").minimize(z_map_loss, var_list=z_var)
        tf.global_variables_initializer().run()



        

        num_optimization_steps = 1000
        losses = []
        print('START')
        feed = { self.view_in: sample_view, self.z: sample_z, self.inputs: sample_image}
        ren_img = self.sess.run(self.G, feed_dict=feed)
        ren_img = inverse_transform(ren_img)
        ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)
        imageio.imwrite(
          os.path.join(
            self.sample_dir, "{0}_samples_{0}.jpg".format(1, 1)),
          ren_img[0])
        imageio.imwrite(
          os.path.join(
            self.sample_dir, "{0}_test_imng_{0}.jpg".format(1, 1)),
          sample_image[0])

        for step in range(num_optimization_steps):
          feed_z_map = { self.view_in: sample_view, self.z: sample_z}
          _, loss = self.sess.run([optimizer, z_map_loss], feed_dict=feed_z_map)
          print('loss: ', loss)

          
          #print('shape: ', gen_img.shape)
        
        print()
        feed = { self.view_in: sample_view, self.z: sample_z}
        ren_img = self.sess.run(self.G, feed_dict=feed)
        ren_img = inverse_transform(ren_img)
        ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)
        imageio.imwrite(
          os.path.join(
            self.sample_dir, "{0}_samples_{0}.jpg".format(2, 2)),
          ren_img[0])

        #print(self.sess.run(z_var))

    def train_HoloGAN(self, config):
       

        self.d_lr_in = tf.compat.v1.placeholder(tf.float32, None, name='d_eta')
        self.g_lr_in = tf.compat.v1.placeholder(tf.float32, None, name='d_eta')

        d_optim = tf.train.AdamOptimizer(cfg['d_eta'], beta1=cfg['beta1'], beta2=cfg['beta2']).minimize(
            self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(cfg['g_eta'], beta1=cfg['beta1'], beta2=cfg['beta2']).minimize(
            self.g_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        shutil.copyfile(sys.argv[1], os.path.join(self.log_dir, 'config.json'))
        self.g_sum = merge_summary([self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary([self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter(self.log_dir, self.sess.graph)


        # Sample noise Z and view parameters to test during training
        sample_z = self.sampling_Z(cfg['z_dim'], str(cfg['sample_z']))
        sample_view = self.gen_view_func(cfg['batch_size'],
                                         cfg['ele_low'], cfg['ele_high'],
                                         cfg['azi_low'], cfg['azi_high'],
                                         cfg['scale_low'], cfg['scale_high'],
                                         cfg['x_low'], cfg['x_high'],
                                         cfg['y_low'], cfg['y_high'],
                                         cfg['z_low'], cfg['z_high'],
                                         with_translation=False,
                                         with_scale=to_bool(str(cfg['with_translation'])))
        sample_files = self.data[0:cfg['batch_size']]
        sample_images = [get_image(sample_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=False) for sample_file in sample_files]

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load()
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        d_lr = cfg['d_eta']
        g_lr = cfg['g_eta']
        for epoch in range(cfg['max_epochs']):
            d_lr = d_lr if epoch < cfg['epoch_step'] else d_lr * (
                cfg['max_epochs'] - epoch) / (cfg['max_epochs'] - cfg['epoch_step'])
            g_lr = g_lr if epoch < cfg['epoch_step'] else g_lr * (
                cfg['max_epochs'] - epoch) / (cfg['max_epochs'] - cfg['epoch_step'])
            random.shuffle(self.data)
            batch_idxs = min(
                len(self.data), config.train_size) // cfg['batch_size']
            batch_idxs = int(batch_idxs)
            for idx in range(0, batch_idxs):
                batch_files = self.data[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
              
                batch_images = [get_image(batch_file,
                                          input_height=self.input_height,
                                          input_width=self.input_width,
                                          resize_height=self.output_height,
                                          resize_width=self.output_width,
                                          crop=self.crop) for batch_file in batch_files]

                batch_z = self.sampling_Z(cfg['z_dim'], str(cfg['sample_z']))
                batch_view = self.gen_view_func(cfg['batch_size'],
                                                cfg['ele_low'], cfg['ele_high'],
                                                cfg['azi_low'], cfg['azi_high'],
                                                cfg['scale_low'], cfg['scale_high'],
                                                cfg['x_low'], cfg['x_high'],
                                                cfg['y_low'], cfg['y_high'],
                                                cfg['z_low'], cfg['z_high'],
                                                with_translation=False,
                                                with_scale=to_bool(str(cfg['with_translation'])))

                feed = {self.inputs: batch_images,
                        self.z: batch_z,
                        self.view_in: batch_view,
                        self.d_lr_in: d_lr,
                        self.g_lr_in: g_lr}
                # Update D network
                _, summary_str = self.sess.run(
                    [d_optim, self.d_sum], feed_dict=feed)
                self.writer.add_summary(summary_str, counter)
                # Update G network
                _, summary_str = self.sess.run(
                    [g_optim, self.g_sum], feed_dict=feed)
                self.writer.add_summary(summary_str, counter)
                # Run g_optim twice
                _, summary_str = self.sess.run(
                    [g_optim, self.g_sum],  feed_dict=feed)
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval(feed)
                errD_real = self.d_loss_real.eval(feed)
                errG = self.g_loss.eval(feed)
                errQ = self.q_loss.eval(feed)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, q_loss: %.8f"
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG, errQ))

                if np.mod(counter, 100) == 1:
                    self.save(counter)
                    feed_eval = {self.inputs: sample_images,
                                 self.z: sample_z,
                                 self.view_in: sample_view,
                                 self.d_lr_in: d_lr,
                                 self.g_lr_in: g_lr}
                    samples, d_loss, g_loss = self.sess.run(
                        [self.G, self.d_loss, self.g_loss],
                        feed_dict=feed_eval)
                    ren_img = inverse_transform(samples)
                    ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)
                    try:
                        imageio.imwrite(
                            os.path.join(
                                self.img_dir, "{0}_GAN.jpg".format(counter)),
                            merge(ren_img, [cfg['batch_size'] // 4, 4]))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" %
                              (d_loss, g_loss))
                    except:
                        imageio.imwrite(
                            os.path.join(
                                self.img_dir, "{0}_GAN.jpg".format(counter)),
                            ren_img[0])
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" %
                              (d_loss, g_loss))

    def sample_HoloGAN(self, config):
        if str.lower(str(cfg["z_map"])) == "true":
            #sample_z = self.z_map
            sample_z = self.sampling_Z(cfg['z_dim'], str(cfg['sample_z']))

        else:
            sample_z = self.sampling_Z(cfg['z_dim'], str(cfg['sample_z']))
            
        could_load, checkpoint_counter = self.load()
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
       
        if config.rotate_azimuth:
            low = cfg['azi_low']
            high = cfg['azi_high']
            step = 45
        elif config.rotate_elevation:
            low = cfg['ele_low']
            high = cfg['ele_high']
            step = 5
        else:
            low = 0
            high = 10
            step = 2

        for i in range(low, high, step):
            if config.rotate_azimuth:
                sample_view = np.tile(
                    np.array([i * math.pi / 180.0, 0 * math.pi / 180.0, 1.0, 0, 0, 0]), (cfg['batch_size'], 1))
            elif config.rotate_azimuth:
                sample_view = np.tile(
                    np.array([270 * math.pi / 180.0, (90 - i) * math.pi / 180.0, 1.0, 0, 0, 0]), (cfg['batch_size'], 1))
            else:
                sample_view = self.gen_view_func(cfg['batch_size'],
                                                 cfg['ele_low'], cfg['ele_high'],
                                                 cfg['azi_low'], cfg['azi_high'],
                                                 cfg['scale_low'], cfg['scale_high'],
                                                 cfg['x_low'], cfg['x_high'],
                                                 cfg['y_low'], cfg['y_high'],
                                                 cfg['z_low'], cfg['z_high'],
                                                 with_translation=False,
                                                 with_scale=to_bool(str(cfg['with_translation'])))

            feed_eval = {self.z: sample_z,
                         self.view_in: sample_view}

            samples = self.sess.run(self.G, feed_dict=feed_eval)
            ren_img = inverse_transform(samples)
            ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)
          
            try:
                imageio.imwrite(
                    os.path.join(
                        self.sample_dir, "{0}_samples_{1}.jpg".format(counter, i)),
                    merge(ren_img, [cfg['batch_size'] // 4, 4]))
            except:
                imageio.imwrite(
                    os.path.join(
                        self.sample_dir, "{0}_samples_{1}.jpg".format(counter, i)),
                    ren_img[0])
          
# =======================================================================================================================

    def sampling_Z(self, z_dim, type="uniform"):
        if str.lower(type) == "uniform":
            return np.random.uniform(-1., 1., (cfg['batch_size'], z_dim))
        else:
            return np.random.normal(0, 1, (cfg['batch_size'], z_dim))

    def linear_classifier(self, features, scope="lin_class", stddev=0.02, reuse=False):
        with tf.compat.v1.variable_scope(scope) as sc:
            w = tf.compat.v1.get_variable('w', [features.get_shape()[-1], 1],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            b = tf.compat.v1.get_variable(
                'biases', 1, initializer=tf.constant_initializer(0.0))
            logits = tf.matmul(features, w) + b
            return tf.nn.sigmoid(logits), logits

    def z_mapping_function(self, z, output_channel, scope='z_mapping', act="relu", stddev=0.02):
        with tf.compat.v1.variable_scope(scope) as sc:
            w = tf.compat.v1.get_variable('w', [z.get_shape()[-1], output_channel * 2],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            b = tf.compat.v1.get_variable('biases', output_channel * 2,
                                initializer=tf.constant_initializer(0.0))
            if act == "relu":
                out = tf.nn.relu(tf.matmul(z, w) + b)
            else:
                out = lrelu(tf.matmul(z, w) + b)
            return out[:, :output_channel], out[:, output_channel:]

# =======================================================================================================================
    def discriminator_IN(self, image,  cont_dim, reuse=False):
        if str(cfg["add_D_noise"]) == "true":
            image = image + tf.random_normal(tf.shape(image), stddev=0.02)

        with tf.compat.v1.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(instance_norm(conv2d_specNorm(
                h0, self.df_dim * 2, name='d_h1_conv'), 'd_in1'))
            h2 = lrelu(instance_norm(conv2d_specNorm(
                h1, self.df_dim * 4, name='d_h2_conv'), 'd_in2'))
            h3 = lrelu(instance_norm(conv2d_specNorm(
                h2, self.df_dim * 8, name='d_h3_conv'), 'd_in3'))

            # Returning logits to determine whether the images are real or fake
            h4 = linear(slim.flatten(h3), 1, 'd_h4_lin')

            # Recognition network for latent variables has an additional layer
            encoder = lrelu((linear(slim.flatten(h3), 128, 'd_latent')))
            cont_vars = linear(encoder, cont_dim, "d_latent_prediction")

            return tf.nn.sigmoid(h4), h4, tf.nn.tanh(cont_vars)

    def generator_AdaIN(self, z, view_in, reuse=False):
        batch_size = tf.shape(z)[0]
        ones = tf.ones([cfg['z_dim'], cfg['z_dim']], tf.float32)
        z_w = tf.Variable(ones, name="z_weight")
        z = tf.matmul(z, z_w)


        s_h, s_w, s_d = 64, 64, 64
        s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(
            s_w, 2), conv_out_size_same(s_d, 2)
        s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(
            s_w2, 2), conv_out_size_same(s_d2, 2)
        s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(
            s_w4, 2), conv_out_size_same(s_d4, 2)
        s_h16, s_w16, s_d16 = conv_out_size_same(
            s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

        with tf.compat.v1.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            # A learnt constant "template"
            with tf.compat.v1.variable_scope('g_w_constant'):
                w = tf.compat.v1.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8],
                                    initializer=tf.random_normal_initializer(stddev=0.02))
                w_tile = tf.tile(tf.expand_dims(
                    w, 0), (batch_size, 1, 1, 1, 1))
                s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
                h0 = AdaIn(w_tile, s0, b0)
                h0 = tf.nn.relu(h0)

            h1 = deconv3d(h0, [batch_size, s_h8, s_w8, s_d8,
                               self.gf_dim * 2], k_h=3, k_d=3, k_w=3, name='g_h1')
            s1, b1 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1')
            h1 = AdaIn(h1, s1, b1)
            h1 = tf.nn.relu(h1)

            h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4,
                               self.gf_dim * 1], k_h=3, k_d=3, k_w=3, name='g_h2')
            s2, b2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2')
            h2 = AdaIn(h2, s2, b2)
            h2 = tf.nn.relu(h2)

            # =============================================================================================================
            h2_rotated = tf_3D_transform(h2, view_in, 16, 16)
            h2_rotated = transform_voxel_to_match_image(h2_rotated)
            # =============================================================================================================
            # Collapsing depth dimension
            h2_2d = tf.reshape(
                h2_rotated, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
            # 1X1 convolution
            h3 = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16],
                          k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3')
            h3 = tf.nn.relu(h3)
            # =============================================================================================================
            h4 = deconv2d(h3, [batch_size, s_h2, s_w2,
                               self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
            s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
            h4 = AdaIn(h4, s4, b4)
            h4 = tf.nn.relu(h4)

            h5 = deconv2d(h4, [batch_size, s_h, s_w,
                               self.gf_dim], k_h=4, k_w=4, name='g_h5')
            s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
            h5 = AdaIn(h5, s5, b5)
            h5 = tf.nn.relu(h5)

            h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim],
                          k_h=4, k_w=4, d_h=1, d_w=1, name='g_h6')

            output = tf.nn.tanh(h6, name="output")
            return output

# =======================================================================================================================
    def save(self, step):
        model_name = "HoloGAN.model"
        self.saver.save(self.sess,
                        os.path.join(self.model_dir, model_name),
                        global_step=step)

    def load(self):
        import re
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(
                self.model_dir, ckpt_name))
            counter = int(
                next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
