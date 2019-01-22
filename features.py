from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Conv2D,Dense, Reshape, BatchNormalization, Activation,Flatten
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt

import cv2
IMAGE_SIZE = 50
# first train with frozen weights, then fine tune
TRAINABLE = False

def create_model(trainable=True):
    model = VGG16(weights='imagenet', include_top=False, input_shape=(50,50, 3))
    x = model.get_layer("block1_conv1").output
    x = Conv2D(1, kernel_size=5, padding="same", strides=1, activation="relu")(x)
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(2, activation="softmax")(x)
    return Model(inputs = model.input, outputs = predictions)

def display_activation(activations, col_size=8, row_size=8):
    activation = activations
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1
def main():
    unscaled = cv2.imread('dois.jpg')
    print(unscaled)
    plt.imshow(unscaled[:,:,:])
    plt.show()
    image = cv2.resize(unscaled, (IMAGE_SIZE, IMAGE_SIZE))
    feat_scaled = preprocess_input(np.array(image, dtype=np.float32))

    model = create_model(trainable=TRAINABLE)
    model.summary()
    activations = model.predict(feat_scaled[np.newaxis,:])
    print(activations.shape)
    #display_activation(activations,1,1)
    #plt.show()
if __name__ == "__main__":
    main()
#cv2.imshow("image", image)
