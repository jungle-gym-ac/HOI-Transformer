python -m torch.distributed.launch \
--nproc_per_node=8 \
--use_env \
main.py \
        --output_dir logs/hico \
        \
        \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        \
        \
        --epochs 150\
        --wandb \
        --backbone resnet50 \
        --pretrained params/detr-r50-pre.pth\
        \
        \
        --enc_layers 3\
        --dec_layers 3
        #--position_embedding 'sine' #‘sine’
        #--no_aux_loss  ######### to be tuned
#wandb sync --clean #