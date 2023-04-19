import json
if __name__=="__main__":
    with open("data/hoi-a/annotations/test_2021.json") as f:
        #"data/v-coco/annotations/test_vcoco.json"
        annotations=json.load(f)
    #print(sum(len(annotation['hoi_annotation']) for annotation in annotations))
    print(len(annotations))
