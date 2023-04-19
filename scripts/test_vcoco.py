from vsrl_eval import VCOCOeval
import os

if __name__ == '__main__':
    vsrl_annot_file="data/v-coco/data/vcoco/vcoco_test.json"
    coco_file="data/v-coco/data/instances_vcoco_all_2014.json"
    split_file="data/v-coco/data/splits/vcoco_test.ids"
    vcocoeval = VCOCOeval(vsrl_annot_file, coco_file, split_file)
    # e.g. vsrl_annot_file: data/vcoco/vcoco_val.json
    #      coco_file:       data/instances_vcoco_all_2014.json
    #      split_file:      data/splits/vcoco_val.ids
    vcocoeval._do_eval("logs/vcoco/vcoco.pickle", ovr_thresh=0.5)