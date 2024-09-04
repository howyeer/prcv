# Copyright (c) Tencent Inc. All rights reserved.
import os
import cv2
import re
import argparse
from PIL import Image
import json
import shutil
import os.path as osp
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# from pycocotools import mask as maskUtils
from mmengine.config import Config, DictAction
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmengine.utils import ProgressBar
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg

from torchvision.ops import nms

import supervision as sv

import sys
sys.path.append('/home/hhy_2023/aaaaacode/YOLO-World')

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=3)
MASK_ANNOTATOR = sv.MaskAnnotator()


class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                 text_scale=0.5,
                                 text_thickness=1)


def xyxy2xywh(xyxy:list):
    x1, y1, x2, y2 = xyxy
    return [x1, y1, x2 - x1, y2 - y1]

def inference_detector(model,
                       image,
                       texts,
                       test_pipeline,
                       max_dets=100,
                       score_thr=0.3,
                       nms_thr=0.5,
                       use_amp=False,
                       task=None):
    W, H  = Image.open(image).size                                               #读取图片，然后获取图片的宽和高
    image_id = image.split('/')[-1].split('.')[0]
    data_info = dict(img_id=0, img_path=image, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    with autocast(enabled=use_amp), torch.no_grad():
        output = model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)       #所有检测框一起做nms
        pred_instances = pred_instances[keep_idxs]


        #----------将面积小的检测框的置信度放大输出----------
        pred_instances = pred_instances[1500<abs(pred_instances.bboxes[:,3]-pred_instances.bboxes[:,1])*   
                                         abs(pred_instances.bboxes[:,2]-pred_instances.bboxes[:,0])]
        
        low_score_pred = pred_instances[pred_instances.scores.float() <= score_thr]
        small_area_pred = low_score_pred[abs(low_score_pred.bboxes[:,3]-low_score_pred.bboxes[:,1])*   
                                         abs(low_score_pred.bboxes[:,2]-low_score_pred.bboxes[:,0])<5000]
        
        small_area_pred = small_area_pred[small_area_pred.scores.float() > 0.1]

        pred_instances = pred_instances[pred_instances.scores.float() >
                                        score_thr]

    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]
    

    pred_instances = pred_instances.to_dict()
    labels = pred_instances.pop('labels') 
    pred_instances['category_id'] = labels               
    for key, value in pred_instances.items():
            pred_instances[key] = value.tolist()
    pred_instances['image_id'] = image_id
    
    #要再次推理的检测框也要处理
    
    small_area_pred = small_area_pred.to_dict()
    labels = small_area_pred.pop('labels') 
    small_area_pred['category_id'] = labels               
    for key, value in small_area_pred.items():
            small_area_pred[key] = value.tolist()
    small_area_pred['image_id'] = image_id
    
    result_perimg = []
    for i in range(len(pred_instances['scores'])):
        xywh = xyxy2xywh(pred_instances['bboxes'][i],W,H)
        result_perimg.append(
            dict(
                image_id=image_id,
                category_id=pred_instances['category_id'][i],
                bbox=xywh,
                score=round(pred_instances['scores'][i],2)))
        
    sarea_infer = []
    for i in range(len(small_area_pred['scores'])):
        xywh = xyxy2xywh(small_area_pred['bboxes'][i],W,H)
        sarea_infer.append(
            dict(
                image_id=image_id,
                category_id=small_area_pred['category_id'][i],
                bbox=xywh,
                score=round(small_area_pred['scores'][i],2)))

    return result_perimg, sarea_infer                                      #返回每张图像的推理结果


def draw_img(val_path,result_list):
    draw_dic = {}
    for result in result_list:
        draw_anno = result['bbox']+[result['score'],result['category_id']]
        if result['image_id'] not in draw_dic:
            draw_dic[result['image_id']] = []
        draw_dic[result['image_id']].append(draw_anno)
    for key,value in draw_dic.items():
        img_path = os.path.join(val_path,key+'.jpg')
        img = cv2.imread(img_path)
        for i in range(len(value)):
            x, y, w, h, score, cat_id = value[i]
            x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, f'{cat_id}, {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 2)
        cv2.imwrite(os.path.join(savedraw_dir,key+'_result.jpg'),img)

        

#对推理完成的结果result.json文件进行计算
def gettaskid(val_path,task):
    taskdic = {}
    taskdic['ebike'] = '0'
    taskdic['carry'] = '1'
    taskdic['mask'] = '2'
    taskdic['hat'] = '3'
    taskdic['head'] = '4'

    imgIds = [imgname for imgname in os.listdir(val_path) if imgname.endswith('.jpg')]
    
    img_list = []
    for imgId in imgIds:
        if imgId[0] == taskdic[task]:
            img_list.append(imgId)
  
    
    return img_list



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO-World Demo')
    parser.add_argument('--config', default="configs/finetune_coco/yolo_world_v2_l_vlpan_bn_sgd__prcv2.py", 
                        help='test config file path')
    parser.add_argument('--checkpoint', default="weights/prcv2.pth",
                        help='checkpoint file')
    parser.add_argument('--image_path', default='data/prcv/val', help='image path,must be a dir.')
    parser.add_argument('--prcv-task', default='mask', help='choose prcv task, including ebike,carry,mask,hat,head')
    parser.add_argument('--text', 
                        default='识别:图中face',
                        help='text prompts, 文本的输入格式为“检测:······”or“识别:······”其中要检测的类别请输入英文,\
                        例如：“检测:图中的face,mask”、"识别:图中是否存在没带口罩的face"、“识别:图中有无carry物品的人”'
                        )
    parser.add_argument('--topk',
                        default=100,
                        type=int,
                        help='keep topk predictions.')
    parser.add_argument('--threshold',
                        default=0.45,
                        type=float,
                        help='confidence score threshold for predictions.')
    parser.add_argument('--nms',
                        default=0.5,
                        type=float,
                        help='nms threshold for predictions.')
    parser.add_argument('--device',
                        default='cuda:0',
                        help='device used for inference.')
    parser.add_argument('--amp',
                        action='store_true',
                        help='use mixed precision for inference.')
    parser.add_argument('--draw',
                        default=False,
                        help='draw the result on the image.')
    parser.add_argument('--output-dir',
                        default='prcv_outputs',
                        help='the directory to save outputs')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
    # init model
    cfg.load_from = args.checkpoint
    model = init_detector(cfg, checkpoint=args.checkpoint, device=args.device)

    # init test pipeline
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    # test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline_cfg)


    #文本的输入格式为“检测：······”or“识别：······”
    if args.text.endswith('.txt'):
        with open(args.text) as f:
            lines = f.readlines()
        texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
    else:
        assert args.text[:3] == '检测:' or args.text[:3] == '识别:'
        text_input = args.text[3:]
        cn_text = re.findall('[\u4e00-\u9fa5]+',args.text)
        for i in range(len(cn_text)):
            text = text_input.replace(cn_text[i],',')
        text = text.strip(',')
        texts = [[t.strip()] for t in text.split(',')] + [[' ']]

    output_dir = args.output_dir
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    task = args.prcv_task
    # load images
    if args.prcv_task is not None:
        assert osp.isdir(args.image_path)
        img_list = gettaskid(args.image_path,args.prcv_task)
        images = [osp.join(args.image_path, img) for img in img_list]
    elif not osp.isfile(args.image):
        images = [
            osp.join(args.image, img) for img in os.listdir(args.image)
            if img.endswith('.png') or img.endswith('.jpg')
        ]
    else:
        images = [args.image]

    # reparameterize texts
    result_list = []
    recog_dic = {}
    model.reparameterize(texts)
    progress_bar = ProgressBar(len(images))
    for image_path in images:
        result_perimg, sarea_infer = inference_detector(model,
                                image_path,
                                texts,
                                test_pipeline,
                                args.topk,
                                args.threshold,
                                args.nms,
                                use_amp=args.amp,
                                task=task)
        
        result_list.extend(result_perimg)
        result_list.extend(sarea_infer)

        if len(result_perimg)==0:
            recog_dic[image_path.split('/')[-1]] = '1'
        else:
            recog_dic[image_path.split('/')[-1]] = '0'

        progress_bar.update()
    

    if args.text[:3] == '检测:':
        with open(os.path.join(output_dir, 'detec.json'), 'w') as f:
            json.dump(result_list, f, indent=4)
    elif args.text[:3] == '识别:':
        with open(os.path.join(output_dir, 'recog.json'), 'w') as f:
            json.dump(recog_dic, f, indent=2)
    
    if args.draw:
        savedraw_dir = os.path.join(output_dir,'visual')
        if os.path.exists(savedraw_dir):
    # 删除整个目录及其内容
            shutil.rmtree(savedraw_dir)
        if not os.path.exists(savedraw_dir):
            os.makedirs(savedraw_dir)
        draw_img(args.image_path,result_list)