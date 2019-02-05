import csv
import math

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Model
import time
#from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import Concatenate, Conv2D, UpSampling2D,MaxPooling2D, Reshape, BatchNormalization, Activation, Input
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K

GRID_SIZE = 112
IMAGE_SIZE = 224

# first train with frozen weights, then fine tune
TRAINABLE = False

EPOCHS = 200
BATCH_SIZE = 4
PATIENCE = 20

MULTI_PROCESSING = True
THREADS = 4

SMOOTH = 1

TRAIN_CSV = "train.csv"
VALIDATION_CSV = "validation.csv"

class DataGenerator(Sequence):

    def __init__(self, csv_file):
        self.paths = []
        self.x1 = []
        self.x2 = []
        self.y1 = []
        self.y2 = []

        with open(csv_file, "r") as file:
            file.seek(0)
            reader = csv.reader(file, delimiter=",")

            for index, row in enumerate(reader):
                for i, r in enumerate(row[1:7]):
                    row[i+1] = int(r)

                path,image_height, image_width, x0, y0, x1, y1 = row

                self.x1.append(np.rint(((GRID_SIZE - 1) / image_width) * x0).astype(int))
                self.x2.append(np.rint(((GRID_SIZE - 1) / image_width) * x1).astype(int))

                self.y1.append(np.rint(((GRID_SIZE - 1) / image_height) * y0).astype(int))
                self.y2.append(np.rint(((GRID_SIZE - 1) / image_height) * y1).astype(int))
                self.paths.append(path)

    def __len__(self):
        return math.ceil(len(self.paths) / BATCH_SIZE)

    def __getitem__(self, idx):
        batch_paths = self.paths[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
        x1 = self.x1[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
        x2 = self.x2[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
        y1 = self.y1[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
        y2 = self.y2[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]

        batch_masks = np.zeros((len(batch_paths), GRID_SIZE, GRID_SIZE))
        batch_images = np.zeros((len(batch_paths), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        for i, f in enumerate(batch_paths):
            img = Image.open(f)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img = img.convert('RGB')
            batch_masks[i, y1[i]: y2[i], x1[i] : x2[i]] = 1
            batch_images[i] = preprocess_input(np.array(img, dtype=np.float32))
            img.close()
        return batch_images, batch_masks[:,:,:,np.newaxis]

class Validation(Callback):
    def __init__(self, generator):
        self.generator = generator

    def on_epoch_end(self, epoch, logs):
        numerator = 0
        denominator = 0

        for i in range(len(self.generator)):
            batch_images, gt = self.generator[i]
            pred = self.model.predict_on_batch(batch_images)

            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0

            numerator += 2 * np.sum(gt * pred)
            denominator += np.sum(gt + pred)

        dice = np.round(numerator / denominator, 4)
        logs["val_dice"] = dice

        print(" - val_dice: {}".format(dice))

def create_model(trainable=True):
    model = VGG16(include_top = False, weights='imagenet', input_shape = (224, 224, 3))

    #model.summary()
    for layer in model.layers:
        layer.trainable = trainable
    block1 = model.get_layer("block1_conv2").output
    block2 = model.get_layer("block2_conv2").output
    block3 = model.get_layer("block3_conv2").output
    poolblock3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block3)
    block4 = Conv2D(256, kernel_size=3, padding="same", strides=1,activation='relu')(poolblock3)

    up1 = UpSampling2D()(block4)
    up1 = Conv2D(128, kernel_size=3, padding="same", strides=1,activation='relu')(up1)
    up1 = Concatenate()([up1, block3])
    up1 = Conv2D(128, kernel_size=3, padding="same", strides=1,activation='relu')(up1)

    up2 = UpSampling2D()(up1)
    up2 = Conv2D(64, kernel_size=3, padding="same", strides=1,activation='relu')(up2)
    up2 = Concatenate()([up2, block2])
    up2 = Conv2D(64, kernel_size=3, padding="same", strides=1,activation='relu')(up2)

    x = Conv2D(1, kernel_size=1, activation="sigmoid")(up2)
    return Model(inputs=model.input, outputs=x)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)

def loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - tf.log(dice_coef(y_true, y_pred))

def main():
    model = create_model(trainable=TRAINABLE)
    model.summary()
    if TRAINABLE:
        model.load_weights(WEIGHTS)

    train_datagen = DataGenerator(TRAIN_CSV)
    validation_datagen = Validation(generator=DataGenerator(VALIDATION_CSV))

    optimizer = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=loss, optimizer=optimizer, metrics=[dice_coef])

    checkpoint = ModelCheckpoint("model-{val_dice:.2f}.h5", monitor="val_dice", verbose=1, save_best_only=True,
                                 save_weights_only=True, mode="max", period=1)
    stop = EarlyStopping(monitor="val_dice", patience=PATIENCE, mode="max")
    reduce_lr = ReduceLROnPlateau(monitor="val_dice", factor=0.2, patience=2, min_lr=1e-9, verbose=1, mode="max")

    model.fit_generator(generator=train_datagen,epochs=EPOCHS,callbacks=[validation_datagen, checkpoint, reduce_lr, stop],workers=THREADS,use_multiprocessing=MULTI_PROCESSING, shuffle=True,verbose=1)
if __name__ == "__main__":
    main()
# object-localization-plateBR
