import tensorflow as tf
from keras import backend as K

K.set_image_data_format('channels_first')

from fr_utils import *
from inception_blocks_v2 import *
# from generator_utils import *
from gen import DataGenerator
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import time
from parameters import *
import keras


def triplet_loss(y_true, y_pred, alpha=ALPHA):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss


best_model_path = None
if os.path.exists("./bestmodel.txt"):
    with open('bestmodel.txt', 'r') as file:
        best_model_path = file.read()

if best_model_path is not None and os.path.exists(best_model_path):
    print("Pre trained model found")
    FRmodel = keras.models.load_model(best_model_path, custom_objects={'triplet_loss': triplet_loss})

else:
    print('Saved model not found, loading untrained FaceNet')
    FRmodel = faceRecoModel(input_shape=(3, IMAGE_SIZE, IMAGE_SIZE))
    load_weights_from_FaceNet(FRmodel)

for layer in FRmodel.layers[0: LAYERS_TO_FREEZE]:
    layer.trainable = False

input_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
A = Input(shape=input_shape, name='anchor')
P = Input(shape=input_shape, name='anchorPositive')
N = Input(shape=input_shape, name='anchorNegative')

enc_A = FRmodel(A)
enc_P = FRmodel(P)
enc_N = FRmodel(N)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=30, min_delta=0.00005)
rd_lr = ReduceLROnPlateau(cooldown=1)
STAMP = 'facenet_%d' % (time.time())
checkpoint_dir = './' + 'checkpoints/'  # + str(int(time.time())) + '/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
cp_callback = ModelCheckpoint(filepath=checkpoint_dir + "ckpt.h5", save_weights_only=False, verbose=1, save_best_only=True)
# Model
tripletModel = Model(inputs=[A, P, N], outputs=[enc_A, enc_P, enc_N])
opt = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
tripletModel.compile(optimizer=opt, loss=triplet_loss)
tripletModel.save(checkpoint_dir + "ckpt.h5")
train_dataset = DataGenerator("./cropped/*/")
valid_dataset = DataGenerator("./cropped_val/*/")
tripletModel.fit_generator(train_dataset,
                           epochs=NUM_EPOCHS,
                           callbacks=[rd_lr, tensorboard, cp_callback],
                           validation_data=valid_dataset
                           )
