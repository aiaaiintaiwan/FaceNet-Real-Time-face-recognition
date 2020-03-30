from keras import backend as K

K.set_image_data_format('channels_first')
from fr_utils import *
from inception_blocks_v2 import *
import keras
from generator_utils import *
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import time
from parameters import *


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
early_stopping = EarlyStopping(monitor='loss', patience=5, min_delta=0.00005)
STAMP = 'facenet_%d' % (len(paths))
checkpoint_dir = './' + 'checkpoints/'  # + str(int(time.time())) + '/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
cp_callback = ModelCheckpoint(filepath=checkpoint_dir + "ckpt.h5", save_weights_only=False, verbose=1)
# Model
tripletModel = Model(inputs=[A, P, N], outputs=[enc_A, enc_P, enc_N])
tripletModel.compile(optimizer='adam', loss=triplet_loss)
tripletModel.save(checkpoint_dir + "ckpt.h5")
gen = batch_generator(BATCH_SIZE)
tripletModel.fit_generator(gen, epochs=NUM_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, callbacks=[early_stopping, tensorboard, cp_callback])
