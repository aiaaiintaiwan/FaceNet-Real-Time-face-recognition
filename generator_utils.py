import glob
import os
import cv2
import numpy as np
from parameters import *
import pickle


class Dataset:
    def __init__(self, path="./cropped/*/"):
        self.input_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
        paths = {}
        for d in glob.glob(path):
            paths[os.path.basename(d[:-1])] = d

        # with open("./path_dict.p", 'rb') as f:
        #     paths = pickle.load(f)
        #     # paths = {}

        # {'001.ALI_HD': 'cropped/001.ALI_HD_right',
        # print(paths)
        self.paths = paths
        faces = []
        for key in paths.keys():
            paths[key] = paths[key].replace("\\", "/")
            faces.append(key)
        self.paths = paths
        self.faces = faces
        images = {}
        for key in paths.keys():
            li = []
            for img in os.listdir(paths[key]):
                img1 = cv2.imread(os.path.join(paths[key], img))
                print(f"reading image {img}         ", end="\r")
                img2 = img1[..., ::-1]
                li.append(np.around(np.transpose(img2, (2, 0, 1)) / 255.0, decimals=12))
            images[key] = np.array(li)
        self.images = images

    def _gen(self, batch_size):
        input_shape = self.input_shape
        y_val = np.zeros((batch_size, 2, 1))
        anchors = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
        positives = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
        negatives = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))

        for i in range(batch_size):
            positive_face = self.faces[np.random.randint(len(self.faces))]
            negative_face = self.faces[np.random.randint(len(self.faces))]
            while positive_face == negative_face:
                negative_face = self.faces[np.random.randint(len(self.faces))]
            anchors[i] = self.images[positive_face][np.random.randint(len(self.images[positive_face]))]
            positives[i] = self.images[positive_face][np.random.randint(len(self.images[positive_face]))]
            negatives[i] = self.images[negative_face][np.random.randint(len(self.images[negative_face]))]

            """            
        while True:
            for i in range(batch_size):
                positive_face = faces[np.random.randint(len(faces))]
                negative_face = faces[np.random.randint(len(faces))]
                while positive_face == negative_face:
                    negative_face = faces[np.random.randint(len(faces))]

                positives[i] = images[positive_face][np.random.randint(len(images[positive_face]))]
                anchors[i] = images[positive_face][np.random.randint(len(images[positive_face]))]
                negatives[i] = images[negative_face][np.random.randint(len(images[negative_face]))]

            x_data = {'anchor': anchors,
                      'anchorPositive': positives,
                      'anchorNegative': negatives
                      }
            print(x_data)
            """
            x_data = {'anchor': anchors,
                      'anchorPositive': positives,
                      'anchorNegative': negatives
                      }
            yield x_data, [y_val, y_val, y_val]

    def batch_generator(self, batch_size=16):
        while True:
            for x in self._gen(batch_size=batch_size):
                yield x

    def calc_steps_pre_epoch(self, batch_size=16):
        count = 0
        for _ in self._gen(batch_size):
            print(f"calc step pre epoch: {count}", end="\r")
            count += 1
        return count
