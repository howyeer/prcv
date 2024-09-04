import os
import json
import cv2
import random
import time
from PIL import Image
import shutil

random.seed(302)

json_save_path ='/data/hhy_2023/prcv/annotations'   #要生成的标准coco格式标签所在文件夹
classes_path ='/data/hhy_2023/prcv/classes.txt'     #类别文件，一行一个类
data_path ='/data/hhy_2023/PRCV_dataset'  #数据集所在文件夹
train_imgpath = '/data/hhy_2023/prcv/train'
val_imgpath = '/data/hhy_2023/prcv/val'

exist_train = os.listdir(train_imgpath)
exist_val = os.listdir(val_imgpath)
for ef in exist_train:
    os.remove(os.path.join(train_imgpath, ef))
for ev in exist_val:
    os.remove(os.path.join(val_imgpath, ev))

task_file = ['ebike','carry','mask','hat','head'] 

def dict_inite():
    with open(classes_path,'r') as fr:                               #打开并读取类别文件
        lines1=fr.readlines()
    # print(lines1)
    categories = []                                                                #存储类别的列表
    for j,label in enumerate(lines1):
        label=label.strip()
        categories.append({'id':j+1,'name':label,'supercategory':'None'})  #将类别信息添加到categories中                                             #将类别信息添加到classes中
    # print(categories)

    write_json_context=dict()                                                      #写入.json文件的大字典
    write_json_context['info']= {'description': '', 'url': '', 'version': '', 'year': 2024, 'contributor': 'howyeer', 'date_created': '2022-07-8'}
    write_json_context['licenses']=[{'id':1,'name':None,'url':None}]
    write_json_context['categories']=categories
    write_json_context['images']=[]
    write_json_context['annotations']=[]

    return write_json_context
 

#接下来的代码主要添加'images'和'annotations'的key值
def get_json_context(imagePath, t_id, new_name, write_json_context):
    # img_pathDir = os.path.join(data_path, tasks[t_id],"images")
    # imageFileList=os.listdir(img_pathDir)                                           #遍历该文件夹下的所有文件，并将所有文件名添加到列表中
    # for i,imageFile in enumerate(imageFileList):
    #     imagePath = os.path.join(img_pathDir,imageFile)                             #获取图片的绝对路径
    image = Image.open(imagePath)                                               #读取图片，然后获取图片的宽和高
    W, H = image.size
    # new_name = str(t_id)+imageFile.strip(tasks[t_id]+'_').zfill(8)

    img_context={}                                                              #使用一个字典存储该图片信息
    #img_name=os.path.basename(imagePath)                                       #返回path最后的文件名。如果path以/或\结尾，那么就会返回空值
    img_context['file_name']=new_name
    img_context['height']=H
    img_context['width']=W
    img_context['date_captured']='2024-07-19'
    img_context['id']=new_name.strip('.jpg')                                                        #该图片的id
    img_context['license']=1
    img_context['color_url']=''
    img_context['flickr_url']=''
    write_json_context['images'].append(img_context)                            #将该图片信息添加到'image'列表中


    txtFile=imagePath.replace('jpg','txt').replace('images','labels_detection')                                               #获取该图片获取的txt文件
    with open(txtFile,'r') as fr:
        lines=fr.readlines()                                                   #读取txt文件的每一行数据，lines2是一个列表，包含了一个图片的所有标注信息
         
        # if len(lines)==0:
            
        

    for j,line in enumerate(lines):

        bbox_dict = {}                                                          #将每一个bounding box信息存储在该字典中
        # line = line.strip().split()
        # print(line.strip().split(' '))

        class_id,x,y,w,h=line.strip().split(' ')                                          #获取每一个标注框的详细信息
        class_id,x, y, w, h = int(class_id), float(x), float(y), float(w), float(h)       #将字符串类型转为可计算的int和float类型

        xmin=(x-w/2)*W                                                                    #坐标转换
        ymin=(y-h/2)*H
        xmax=(x+w/2)*W
        ymax=(y+h/2)*H
        x=x*W
        y=y*H
        w=w*W
        h=h*H

        bbox_dict['id']=int(new_name.strip('.jpg'))*100+j                                           
                    #bounding box的坐标信息
        bbox_dict['image_id']=new_name.strip('.jpg')  
        #------------------------------------------------------------------------------------------------------------------
        if t_id >= 3:
            bbox_dict['category_id']=t_id+class_id+1+1                                              #注意目标类别要加一
        else:
            bbox_dict['category_id']=t_id+class_id+1
        #------------------------------------------------------------------------------------------------------------------
        bbox_dict['iscrowd']=0
        height,width=abs(ymax-ymin),abs(xmax-xmin)
        bbox_dict['area']=height*width
        bbox_dict['bbox']=[xmin,ymin,w,h]
        bbox_dict['segmentation']=[]
        write_json_context['annotations'].append(bbox_dict)                               #将每一个由字典存储的bounding box信息添加到'annotations'列表中
      

def split_trainval(data_path, train_imgpath, val_imgpath, t_id, tasks, ratio=0.8):
    img_pathDir = os.path.join(data_path, tasks[t_id], "images")
    img_list = os.listdir(img_pathDir)
    img_num = len(img_list)
    train_num = int(img_num * ratio)
    num_ls = list(range(img_num))
    random.shuffle(num_ls)
    
    for i in range(train_num):
        image_path = os.path.join(img_pathDir, img_list[num_ls[i]])
        new_name = str(t_id)+img_list[num_ls[i]].strip(tasks[t_id]+'_').zfill(8)  
        new_path = os.path.join(train_imgpath, new_name)
        shutil.copy(image_path, new_path)
        
        get_json_context(image_path, t_id, new_name, train_json_context)
        
        

    
    for i in range(train_num, img_num):
        image_path = os.path.join(img_pathDir, img_list[num_ls[i]])
        new_name = str(t_id)+img_list[num_ls[i]].strip(tasks[t_id]+'_').zfill(8) 
        new_path = os.path.join(val_imgpath, new_name)
        shutil.copy(image_path, new_path)
        
        get_json_context(image_path, t_id, new_name, val_json_context)
        
        

train_json_context = dict_inite()
val_json_context = dict_inite()

for i in range(len(task_file)): 
    split_trainval(data_path, train_imgpath, val_imgpath, i, task_file, ratio=0.9992)
    

name = os.path.join(json_save_path,"train"+'.json')
with open(name,'w') as fw:                                                                #将字典信息写入.json文件中
    json.dump(train_json_context,fw,indent=2)

name = os.path.join(json_save_path,"val"+'.json')
with open(name,'w') as fw:                                                                #将字典信息写入.json文件中
    json.dump(val_json_context,fw,indent=2)

# name = os.path.join(json_save_path,"all_data"+'.json')
# with open(name,'w') as fw:                                                                #将字典信息写入.json文件中
#     json.dump(write_json_context,fw,indent=2)

recognition_labels = {}
for t_id in range(len(task_file)):
    img_pathDir = os.path.join(data_path, task_file[t_id], "images") 
    img_list = os.listdir(img_pathDir)
    for name in img_list:
        new_name = str(t_id)+name.strip(task_file[t_id]+'_').zfill(8)
        label_path = os.path.join(img_pathDir.replace('images','labels_recognition'),name.replace('jpg','txt'))
        with open(label_path,'r') as lf:
            label_rec = lf.readline()
            label_rec = label_rec.strip()
        recognition_labels[new_name] = label_rec

name = os.path.join(json_save_path,"recog_labels"+'.json')
with open(name,'w') as fw:                                                                #将字典信息写入.json文件中
    json.dump(recognition_labels,fw,indent=2)