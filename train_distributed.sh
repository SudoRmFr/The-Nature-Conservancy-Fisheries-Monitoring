# usage: ./train_distributed.sh fold_no
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 train_distributed.py --fold "$1"