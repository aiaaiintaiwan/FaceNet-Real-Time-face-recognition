import glob
import math
import os
import random

import cv2
import tensorflow.keras as keras
import numpy as np

from parameters import IMAGE_SIZE


class DataGenerator(keras.utils.Sequence):

    def __init__(self, path, batch_size=64, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batches_index = []  # [N][3(arch, pos, neg)]
        self.input_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
        paths = {}
        for d in glob.glob(path):
            paths[os.path.basename(d[:-1])] = d

        faces = []
        for key, v in paths.items():
            paths[key] = paths[key].replace("\\", "/")
            faces.append(key)
        self.paths = paths
        self.faces = faces
        images = {}
        for key in paths.keys():
            li = []
            for img in os.listdir(paths[key]):
                img1 = cv2.imread(os.path.join(paths[key], img))
                print(f"reading image {img}                 ", end="\r")
                img2 = img1[..., ::-1]
                li.append(np.around(np.transpose(img2, (2, 0, 1)) / 255.0, decimals=12))
            images[key] = np.array(li)
        self.images = images

        # generate BATCH
        for cls in faces:
            for anc_img_id in range(len(images[cls])):
                anc_img = (cls, anc_img_id)
                for neg_cls in faces:
                    if neg_cls == cls:
                        continue
                    pos_img = (cls, np.random.randint(len(self.images[cls])))
                    neg_img = (neg_cls, np.random.randint(len(self.images[neg_cls])))
                    self.batches_index.append([anc_img, pos_img, neg_img])

    def __len__(self):
        # 计算每一个epoch的迭代次数
        return math.ceil(len(self.batches_index) / float(self.batch_size))

    def __getitem__(self, index):
        # 生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        ancs = poses = negs = []

        for i in range(index * self.batch_size, index * self.batch_size + self.batch_size):
            if i >= len(self.batches_index): break
            (pos_cls, anc_id), (_, pos_id), (neg_cls, neg_id) = self.batches_index[i]
            anc = self.images[pos_cls][anc_id]
            pos = self.images[pos_cls][pos_id]
            neg = self.images[neg_cls][neg_id]
            ancs.append(anc)
            poses.append(pos)
            negs.append(neg)
        return [np.asarray(ancs), np.asarray(poses), np.asarray(negs)], [np.asarray(ancs), np.asarray(poses), np.asarray(negs)]

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle:
            random.shuffle(self.batches_index)
