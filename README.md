由于对通识大模型的训练在检测任务中表现不够好，所以本队最终使用的代码不是通识大模型，而是有open-vocabulary能力的算法模型-YOLO-World。\
最终模型也能完成本次比赛的所有任务，并且按照设置提示词输入也可对本次比赛之外的任务进行检测和识别。例如demo_out文件中的图片。\
代码中prcv_txt.py可以复现对测试集任务的检测和识别。输入的提示词在text参数中有说明，task任务是用于选择不同任务文件夹的，不算模型输入。\
使用prcv_txt.py也可以对任意图片进行检测和识别。只需要更改文件保存路径out_dir参数即可。\
threshold参数用于调节置信度阈值

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
