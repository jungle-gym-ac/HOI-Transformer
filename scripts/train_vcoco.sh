python -m torch.distributed.launch \
--nproc_per_node=8 \
--use_env \
main.py \
        --output_dir logs/vcoco \
        \
        \
        --dataset_file vcoco \
        --hoi_path data/v-coco \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        \
        \
        --set_cost_bbox 2.5 \
        --set_cost_giou 1 \
        --bbox_loss_coef 2.5 \
        --giou_loss_coef 1\
        \
        \
        --epochs 150\
        --wandb\
        --backbone resnet50 \
        --pretrained params/detr-r50-pre-vcoco.pth \
        \
        \
        #--no_aux_loss
        #--position_embedding 'sine'\
          ######### to be tuned
wandb sync --clean #

python generate_vcoco_official.py \
        --param_path logs/vcoco/checkpoint.pth \
        --save_path logs/vcoco/vcoco.pickle \
        --hoi_path data/v-coco

#test on vcoco test set
PYTHONPATH=data/v-coco \
    python "scripts/test_vcoco.py"