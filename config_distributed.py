from datetime import datetime
##################################################
# Training Config
##################################################
workers = 1                 # number of Dataloader workers
epochs = 50                # number of epochs
batch_size = 24             # batch size
learning_rate = 1e-3        # initial learning rate

##################################################
# Model Config
##################################################
image_size = (224, 224)     # size of training images
net = 'resnet50'  # feature extractor
num_attentions = 32     # number of attention maps
beta = 5e-2                 # param for update feature centers

##################################################
# Dataset/Path Config
##################################################
tag = 'fish'                # 'aircraft', 'bird', 'car', or 'dog'

# saving directory of .ckpt models
save_dir = f'./save/{datetime.now().strftime("%Y_%m-%d_%H:%M")}-{net}-{image_size[0]}-{image_size[1]}-{num_attentions}-{batch_size}/'
model_name = 'model.ckpt'
log_name = 'train.log'

# checkpoint model for resume training
ckpt = False
# ckpt = save_dir + model_name