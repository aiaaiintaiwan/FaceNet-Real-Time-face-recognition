"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyleft (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the lefts
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyleft notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYleft HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import detect_face
import random
from time import sleep
import pickle
from parameters import *


def main(args):
    sleep(random.random())
    output_dir = os.path.expanduser(args.output_dir)
    path_dict = {}
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(args.input_dir)

    print('Creating networks and loading parameters')

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
    cascade = cv2.CascadeClassifier("fd_models/haarcascade_mcs_leftear.xml")
    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        if args.random_order:
            random.shuffle(dataset)
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name + "_left")
            path_dict[cls.name] = output_class_dir
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                if args.random_order:
                    random.shuffle(cls.image_paths)
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename + '.png')
                print(image_path)
                if not os.path.exists(output_filename):
                    try:
                        img = misc.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim < 2:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[:, :, 0:3]

                        if not args.crop:
                            print("no crop")
                            img = cv2.resize(img, (args.image_size, args.image_size))
                            filename_base, file_extension = os.path.splitext(output_filename)
                            if args.detect_multiple_faces:
                                output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                            else:
                                output_filename_n = "{}{}".format(filename_base, file_extension)
                            misc.imsave(output_filename_n, img)
                            nrof_successfully_aligned += 1
                        else:
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                            # Detect object in input video stream
                            objects = cascade.detectMultiScale(
                                gray,
                                scaleFactor=1.1,
                                minNeighbors=5,
                                minSize=(30, 30),
                                flags=cv2.CASCADE_SCALE_IMAGE
                            )

                            # Iterate over each face found
                            for i, (x, y, w, h) in enumerate(objects):
                                # det = np.squeeze(det)
                                bb = np.zeros(4, dtype=np.int32)
                                bb[0] = np.maximum(x - args.margin / 2, 0)
                                bb[1] = np.maximum(y - args.margin / 2, 0)
                                bb[2] = np.minimum((x + w) + args.margin / 2, gray.shape[1])
                                bb[3] = np.minimum((y + h) + args.margin / 2, gray.shape[0])
                                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                                scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')

                                filename_base, file_extension = os.path.splitext(output_filename)
                                if args.detect_multiple_faces:
                                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                else:
                                    output_filename_n = "{}{}".format(filename_base, file_extension)
                                misc.imsave(output_filename_n, scaled)
                                nrof_successfully_aligned += 1
                                text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

    with open('path_dict.p', 'wb') as f:
        pickle.dump(path_dict, f)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--crop', type=int, help='Directory with unaligned images.')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=IMAGE_SIZE)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order',
                        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
