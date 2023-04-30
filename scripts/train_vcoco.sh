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
        --epochs 150\
        --wandb\
        --backbone resnet50 \
        --pretrained params/detr-r50-pre-vcoco.pth \
        \
        \
        --enc_layers 6\
        --dec_layers 6
        #--no_aux_loss
        #--position_embedding 'sine'\
wandb sync --clean #

python generate_vcoco_official.py \
        --param_path logs/vcoco/checkpoint.pth \
        --save_path logs/vcoco/vcoco.pickle \
        --hoi_path data/v-coco

#test on vcoco test set
PYTHONPATH=data/v-coco \
    python "scripts/test_vcoco.py"