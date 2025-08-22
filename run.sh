export OMP_NUM_THREADS=16

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_v2.py configs/ms1mv2_vit_s_qs.py --mode token

