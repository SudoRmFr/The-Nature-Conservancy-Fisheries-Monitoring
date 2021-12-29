from datetime import datetime

##################################################
# Training Config
##################################################
GPU = '0'  # GPU
workers = 4  # number of Dataloader workers
epochs = 30  # number of epochs
batch_size = 64  # batch size
learning_rate = 1e-3  # initial learning rate

##################################################
# Model Config
##################################################
image_size = (224, 224)  # size of training images
net = 'resnet50'  # feature extractor
num_attentions = 32  # number of attention maps
beta = 5e-2  # param for update feature centers

visual_path = './visual'  # Where to save the visualized image when running infer.py

##################################################
# Dataset/Path Config
##################################################
tag = 'fish'  # 'aircraft', 'bird', 'car', or 'dog'

# checkpoint model for resume training
ckpt = './save/???'
