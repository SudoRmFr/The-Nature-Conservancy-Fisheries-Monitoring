from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import pandas as pd
import numpy as np
import shutil
import math
import cv2
import sys
import os

def IcptV3_T(include_top=False, input_shape=None, pooling="avg", classes=1000):
    from tensorflow.keras.applications import InceptionV3
    bb = InceptionV3(
        include_top=include_top, weights="imagenet",
        input_tensor=None, input_shape=input_shape
    )

    bb_out = bb.get_layer(index=-1).output
    outputs = layers.AveragePooling2D((8, 8), strides=(8, 8), name="avg_pool")(bb_out)
    outputs = layers.Flatten(name="flatten")(outputs)
    outputs = layers.Dense(classes, activation="softmax", name="predictions")(outputs)

    return models.Model(inputs=bb.input, outputs=outputs, name="icptv3")

from tensorflow.keras.applications.inception_v3 import preprocess_input
# def preprocess_input(x):
#     PIXEL_MEAN = [96.48, 107.20, 99.98]
#     return (x - PIXEL_MEAN) / 255.0

class BestModelCkpt(callbacks.Callback):
    def __init__(self, filepath="./", monitor="val_loss", mode="min", verbose=1):
        super(BestModelCkpt, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        if mode == "min":
            self.mode = mode; self.bound = np.inf
        elif mode == "max":
            self.mode = mode; self.bound = -np.inf
        else:
            raise ValueError("mode in BestModelCkpt can only be \'min\' or \'max\'")
        self.verbose = verbose
        self.ckptlog = ""
    
    def on_epoch_end(self, epoch, logs=None):
        if ((self.mode == "min" and logs[self.monitor] < self.bound) or \
            (self.mode == "max" and logs[self.monitor] > self.bound)):
            self.ckptlog = "Epoch {}: {} improved from {:.6f} to {:.6f}\n".format(
                epoch, self.monitor, self.bound, logs[self.monitor]
            )
            if self.verbose:
                print(self.ckptlog)
            self.bound = logs[self.monitor]
            self.model.save(self.filepath)
    
    def on_train_end(self, logs=None):
        print("SAVE {}: {}".format(self.filepath, self.ckptlog))

def format_dd(mode="form"):
    label_names = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]
    if mode == "form":
        for label in label_names:
            dirname = os.path.join("train", label)
            img_names = os.listdir(dirname)
            train_imgs, valid_imgs = train_test_split(img_names, test_size=0.2)
            outpath = os.path.join("valid", label)
            os.makedirs(outpath, exist_ok=True)
            for img_name in valid_imgs:
                shutil.move(os.path.join(dirname, img_name), outpath)
    else:
        for label in label_names:
            dirname = os.path.join("valid", label)
            img_names = os.listdir(dirname)
            outpath = os.path.join("train", label)
            for img_name in img_names:
                shutil.move(os.path.join(dirname, img_name), outpath)
    #return

"""
format_dd("back")
exit(0)
"""

#FISH_CLASSES = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]
EPOCHS      = 100
BATCH_SIZE  = 64
WEIGHTS     = None
OUTPUT_PATH = "models"
ROW = 300
COL = 300
os.makedirs(OUTPUT_PATH, exist_ok=True)

# datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    zoom_range=0.2, channel_shift_range=20,
    horizontal_flip=True, vertical_flip=True,
)

format_dd("form")
trainset = datagen.flow_from_directory("train", target_size=(ROW, COL), batch_size=BATCH_SIZE)
validset = datagen.flow_from_directory("valid", target_size=(ROW, COL), batch_size=BATCH_SIZE)

if WEIGHTS is None:
    model = IcptV3_T(input_shape=(ROW, COL, 3), classes=8)
    # model = DB_Model((224, 224, 3), 8)
else:
    model = models.load_model(WEIGHTS)
model.summary()
opt = optimizers.SGD(lr=1e-4, momentum=0.9, decay=0.0, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])
ckptlist = [
    BestModelCkpt(
        filepath=os.path.join(OUTPUT_PATH, "icptv3-fish.h5"),
        #filepath=os.path.join(OUTPUT_PATH, "dbmodel-fish.h5"), 
        monitor="val_acc", mode="max", verbose=1
    )
]
history = model.fit(
    trainset, epochs=EPOCHS, verbose=2, callbacks=ckptlist, validation_data=validset
)

format_dd("back")
