from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time
import cv2
import csv
import sys
import os


from tensorflow.keras.applications.inception_v3 import preprocess_input
#def preprocess_input(x):
#    PIXEL_MEAN = [96.48, 107.20, 99.98]
#    return (x - PIXEL_MEAN) / 255.0


# trainset = datagen.flow_from_directory("train", target_size=(224, 224), batch_size=32)
# print(trainset.class_indices)

BATCH_SIZE  = 64
WEIGHTS     = "models/icptv3-fish.h5"
ROW = 300
COL = 300

model = models.load_model(WEIGHTS)
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    zoom_range=0.2, channel_shift_range=20,
    horizontal_flip=True, vertical_flip=True
)

with open("submission.csv", "w", newline="") as csvFD:
    writer = csv.writer(csvFD)
    writer.writerow(["image", "ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"])
    
    pred = None
    img_names = None
    n_aug = 10
    for i in range(n_aug):
        TIME_STAMP = time.time()
        test_generator = datagen.flow_from_directory(
            "test", target_size=(ROW, COL), 
            batch_size=BATCH_SIZE,
            shuffle=False, # Important !!!
            classes=None, class_mode=None
        )
        if pred is None:
            pred = model.predict(test_generator)
            img_names = test_generator.filenames
        else:
            pred += model.predict(test_generator)
        print("Round {} Done: {:.2f} sec".format(i + 1, time.time() - TIME_STAMP))
    pred /= n_aug

    for i in range(len(img_names)):
        row = [os.path.basename(img_names[i])]
        if "test_stg2" in img_names[i]:
            row = ["test_stg2/{}".format(os.path.basename(img_names[i]))]

        for entry in pred[i]:
            row.append("{:.8f}".format(entry))
        writer.writerow(row)
print("Generate submission.csv [DONE]")

