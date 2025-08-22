export OMP_NUM_THREADS=16

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_v2_guray.py configs/casia_vit_s_qs.py --mode token

