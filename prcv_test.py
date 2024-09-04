from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
import json
import copy
from pycocotools import mask as maskUtils

task = 'mask'

class COCOeval_custom(COCOeval): 
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        super(COCOeval_custom,self).__init__(cocoGt, cocoDt, iouType)
        self.issue_ids = set()

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        if  len(g) != len(d):
            self.issue_ids.add(imgId+'_len')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
    
        for i in range(len(ious)):
            if max(ious[i])<0.5:
                self.issue_ids.add(imgId+'_iou')

        return ious


# Load the COCO dataset
dataDir = 'data/prcv/val'
annFile = 'data/prcv/annotations/val.json'
result = 'prcv_outputs/'
cocoGt = COCO(annFile)

imgIds_dic = dict()
imgIds_dic['ebike'] = []
imgIds_dic['carry'] = []
imgIds_dic['mask'] = []
imgIds_dic['hat'] = []
imgIds_dic['head'] = []

imgIds = cocoGt.getImgIds()

for imgId in imgIds:
    if imgId[0] == '0':
        imgIds_dic['ebike'].append(imgId)
    elif imgId[0] == '1':
        imgIds_dic['carry'].append(imgId)
    elif imgId[0] == '2':
        imgIds_dic['mask'].append(imgId)
    elif imgId[0] == '3':
        imgIds_dic['hat'].append(imgId)
    elif imgId[0] == '4':
        imgIds_dic['head'].append(imgId)

# # for task in imgIds_dic.keys():
# #     for imgId in imgIds_dic[task].keys():
# #         annids = cocoGt.getAnnIds(imgIds=[imgId])
# #         anns = cocoGt.loadAnns(annids)
# #         anns_list = []
# #         for ann in anns:
# #             ann_info = copy.deepcopy(ann['bbox'])
# #             ann_info.insert(0,ann['category_id'])
# #             anns_list.append(ann_info)
# #         imgIds_dic[task][imgId]['file_name'] = cocoGt.loadImgs([imgId])[0]['file_name']
# #         imgIds_dic[task][imgId]['height'] = cocoGt.loadImgs([imgId])[0]['height']
# #         imgIds_dic[task][imgId]['width'] = cocoGt.loadImgs([imgId])[0]['width']
# #         imgIds_dic[task][imgId]['anns'] = anns_list

# # with open('./try.json','w') as f:
# #     json.dump(imgIds_dic,f,indent=4)




cocoDt = cocoGt.loadRes(os.path.join(result,task + '_detec.json'))
cocoEval = COCOeval_custom(cocoGt, cocoDt, 'bbox')
cocoEval.params.imgIds = imgIds_dic[task]

cocoEval.evaluate()
cocoEval.accumulate()
#----------------------summarize----------------------
p = cocoEval.params
aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == 'all']
mind = [i for i, mDet in enumerate(p.maxDets) if mDet == 100]
s = cocoEval.eval['precision']
t = np.where(0.5 == p.iouThrs)[0]
s = s[t]
s = s[:,:,:,aind,mind]
if len(s[s>-1])==0:
    mean_s = -1
else:
    mean_s = np.mean(s[s>-1])
print(f'{task} detection mAP: @IoU=0.5 {mean_s}'.format(mean_s))
# cocoEval.summarize()
print(cocoEval.issue_ids)



recoglabels_path = 'data/prcv/annotations/recog_labels.json'
recog_results_path = os.path.join(result, task+'_recog.json')

issue_ids = []
tp = 0
recall_0 = 0
precision_0 = 0

with open(recog_results_path) as f:
    recog_results = json.load(f)
with open(recoglabels_path) as lf:
    recog_labels = json.load(lf)


for result in recog_results.keys():
    if recog_results[result] == recog_labels[result] == '0':
        tp += 1
    if recog_labels[result] == '0':
        recall_0 += 1
    if recog_results[result] == '0':
        precision_0 += 1
    if recog_results[result] != recog_labels[result]:
        issue_ids.append(result)


precision = tp/precision_0
recall = tp/recall_0
F1_0 = 2*precision*recall/(precision+recall)  
print(precision)
print(recall)
print(f'{task} recognition F1: {F1_0}'.format(F1_0))
# print(issue_ids)