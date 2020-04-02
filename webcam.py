import glob
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from os.path import isfile
import tensorflow as tf
from tensorflow.keras.models import load_model

from fr_utils import *
from inception_blocks_v2 import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

K.set_image_data_format('channels_first')
PADDING = 10
ready_to_detect_identity = True
THRESHOLD = 1.
# FRmodel = faceRecoModel(input_shape=(3, 96, 96))
if not isfile("checkpoints/facenet.h5"):
    full_model: Model = load_model("checkpoints/ckpt.h5", compile=False)
    fr_model = Model(inputs=full_model.get_layer("FaceRecoModel").inputs,
                     outputs=full_model.get_layer("FaceRecoModel").outputs)
    fr_model.save("checkpoints/facenet.h5")
else:
    fr_model = load_model("checkpoints/facenet.h5", compile=False)

fr_model.summary()


def triplet_loss(y_true, y_pred, alpha=0.3):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: 計算anchor和positive的編碼(距離)
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: 計算anchor和negative的編碼(距離)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: 將先前計算出的距離相減並加上邊距alpha
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: 將上述計算出的損失與零取最大值，再將所有樣本加總起來
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss


fr_model.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])


# load_weights_from_FaceNet(FRmodel)


def img_path_to_encoding(image_path, fr_model):
    img1 = cv2.imread(image_path, 1)
    return img_to_encoding(img1, fr_model)


def prepare_database():
    database = {}

    # load all the images of individuals to recognize into the database
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_path_to_encoding(file, fr_model)

    return database


def webcam_face_recognizer():
    """
    Runs a loop that extracts images from the computer's webcam and determines whether or not
    it contains the face of a person in our database.

    If it contains a face, an audio message will be played welcoming the user.
    If not, the program will process the next frame from the webcam
    """
    global ready_to_detect_identity

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('./fd_models/haarcascade_frontalface_default.xml')
    frame_rate = 60
    prev = 0
    while vc.isOpened():
        time_elapsed = time.time() - prev
        _, frame = vc.read()
        # frame = cv2.imread("/home/ray1422/Desktop/test.jpg")
        if time_elapsed > 1. / frame_rate:
            prev = time.time()

            img = frame

            # We do not want to detect a new identity while the program is in the process of identifying another person
            if ready_to_detect_identity:
                img = process_frame(img, frame, face_cascade)
                cv2.imshow("preview", img)

        key = cv2.waitKey(100)

        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow("preview")


def process_frame(img, frame, face_cascade):
    """
    Determine whether the current frame contains the faces of people from our database
    """
    global ready_to_detect_identity
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # , 1.3, 5

    # Loop through all the faces detected and determine whether or not they are in the database
    identities = []
    for (x, y, w, h) in faces:
        x1 = x - PADDING
        y1 = y - PADDING
        x2 = x + w + PADDING
        y2 = y + h + PADDING

        img = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        identity = find_identity(frame, x1, y1, x2, y2)

        if identity is not None:
            # identities.append(identity)  # 把who_is_it回傳的人名家到字典"identities"裡面
            img = cv2.rectangle(frame, (x1 - 10, y1 - 30), (x1 + 90, y1 + 10), (0, 0, 0), cv2.FILLED)
            img = cv2.putText(img, identity, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1,
                              cv2.LINE_AA)
    # if identities:  # 如果字典有人名
    # #     cv2.imwrite('example.png', img)
    #
    #     ready_to_detect_identity = False
    #     pool = Pool(processes=1)
    #     # We run this as a separate process so that the camera feedback does not freeze
    #     pool.apply_async(welcome_users, [identities])
    return img


def find_identity(frame, x1, y1, x2, y2):
    """
    Determine whether the face contained within the bounding box exists in our database

    x1,y1_____________
    |                 |
    |                 |
    |_________________x2,y2

    """
    height, width, channels = frame.shape
    # The padding is necessary since the OpenCV face detector creates the bounding box around the face and not the head
    part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]

    return who_is_it(part_image, database, fr_model)


def who_is_it(image, _database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.

    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    pretrained_model -- your Inception pretrained_model instance in Keras

    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    encoding = img_to_encoding(image, model)

    min_dist = 100
    identity = None

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in _database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding)

        print('distance for %s is %s' % (name, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > THRESHOLD:
        return None
    else:
        return str(identity)


def welcome_users(identities):
    """ Outputs a welcome audio message to the users """
    global ready_to_detect_identity
    welcome_message = 'Welcome '

    if len(identities) == 1:
        welcome_message += '%s, have a nice day.' % identities[0]
        # print('Welcome %s, have a nice day!' %(identities[0]))
    else:
        for identity_id in range(len(identities) - 1):
            welcome_message += '%s, ' % identities[identity_id]
        welcome_message += 'and %s, ' % identities[-1]
        welcome_message += 'have a nice day!'

    # windows10_voice_interface.Speak(welcome_message)
    print(welcome_message)

    # Allow the program to start detecting identities again
    ready_to_detect_identity = True


if __name__ == "__main__":
    database = prepare_database()
    webcam_face_recognizer()

# ### References:
#
# - Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
# - Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf)
# - The pretrained pretrained_model we use is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
# - Our implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet
#
