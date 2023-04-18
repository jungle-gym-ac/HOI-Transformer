import json
if __name__=="__main__":
    with open("data/HOI-A/annotations/test_2019.json") as f:
        #"data/v-coco/annotations/test_vcoco.json"

        annotations=json.load(f)
    #print(sum(len(annotation['hoi_annotation']) for annotation in annotations))
    print(len(annotations))
