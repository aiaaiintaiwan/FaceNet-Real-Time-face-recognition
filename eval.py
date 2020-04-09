import glob
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow_core.python.keras.models import load_model
from PIL import Image

from webcam import img_path_to_encoding, who_is_it

DIR_NAME = "./cropped_val"
MODEL_PATH = "./checkpoints/facenet.h5"
model = load_model(MODEL_PATH)


def main():
    confusions = np.zeros((5, 5), dtype=np.int32)
    tests = {}
    database = {}
    anc = {}
    total = 0
    correct = 0

    class2id = {}
    id2class = {}
    # prepare database
    for i, c in enumerate(glob.glob(DIR_NAME + "/*/")):
        if os.path.isdir(c):
            class_name = os.path.basename(os.path.dirname(c))
            class2id[class_name] = i
            id2class[i] = class_name
            flag = True
            for f in glob.glob(c + "/*"):
                if flag:
                    flag = False
                    tests[class_name] = []
                    anc[class_name] = f
                    print(class_name)
                    database[class_name] = img_path_to_encoding(f, model)
                else:
                    tests[class_name].append(f)

    plt.figure(figsize=(10, 5))  # 设置窗口大小
    # start eval
    for class_name, fs in tests.items():
        total = correct = 0
        for f in fs:
            img = cv2.imread(f)
            img = cv2.resize(img, (96, 96))
            predict = who_is_it(img, database, model)
            total += 1
            if predict == class_name:
                correct += 1
            # print(f"predict: {predict} real: {class_name}, {predict == class_name} Accuracy: {correct / total:.3f}")
            confusions[class2id[class_name], class2id[predict]] += 1
        print(f"{class_name} Acc: {correct / total:.4f}")

        # plt.suptitle('Multi_Image')  # 图片名称
        plt.subplot(2, 3, class2id[class_name]+1), plt.title(f'{class_name.replace("_right", "")} {correct / total:.4f}')
        img = Image.open(os.path.join(anc[class_name]))
        plt.imshow(img), plt.axis('off')
    print(confusions)
    plt.show()

if __name__ == '__main__':
    main()
