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
        --set_cost_bbox 2.5 \
        --set_cost_giou 1 \
        --bbox_loss_coef 2.5 \
        --giou_loss_coef 1\
        \
        \
        --epochs 150\
        --wandb \
        --backbone resnet50 \
        --pretrained params/detr-r50-pre.pth \
        \
        --position_embedding 'learned' #‘sine’
        #--no_aux_loss  ######### to be tuned
wandb sync --clean #