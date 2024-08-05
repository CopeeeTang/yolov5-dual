# yolov5-dual
 ## A model that achieve dual detection(Infrared+RGB) with rotation

训练数据集为天津大学开源的VisDoone数据集 在旋转框检测的基础上融合红外检测，实现双模态目标检测。
### 该分支为红外检测
### 旋转框目标检测详情请见另一个分支 yolo-obb-master

### 此外我们复现了将红外图片还原成可见光图片的工作

./weights 为权重文件 包括训练好的dual.pt格式及经过推理生成的dual.onnx格式
./runs 为检测及训练的结果 我们在本地跑了200epochs
trai.py 为训练文件 dualdetechchange.py 为双模态检测文件

该项目尚未完善 待后续开发 目前实现了红外检测