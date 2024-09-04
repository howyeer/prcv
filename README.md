## 复现模型推理
比赛测试内容复现：使用prcv_txt.py，其中模型文件下载链接为：[prcv2.pth](https://huggingface.co/howyeer/prcv/blob/main/prcv2.pth)存放在weights文件夹下
测试数据集放在data/prcv_test文件夹下，文件目录形式为
```bash
├── data
│   ├── coco
│   ├── prcv_test
│   │   ├── carry
│   │   │    ├── images_test
│   │   ├── ebike
│   │   │    ├── images_test
│   │   ├── hat
│   │   │    ├── images_test
│   │   ├── head
│   │   │    ├── images_test
│   │   ├── mask
│   │   │    ├── images_test
```
运行脚本
```bash
python prcv_txt.py --prcv_task "任务文件夹(例如carry)" --text "提示词输入(请根据具体格式进行输入)"  --output_dir "输出文件夹(例如carry_out)"

```
**注意**:这里prcv_task只是测试数据集时选择需要测试的文件夹，并不是模型的输入


## 进行通识能力测试
通识能力测试使用的是prcv_infer.py
```bash
python prcv_infer.py --image_path '输入图像路径' --prcv_task None --text "提示词输入(请根据具体格式进行输入)" --output_dir "输出文件夹"
```
这里的task输入为None
