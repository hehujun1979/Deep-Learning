 **数据处理：** 
    首先将使用Arcgis平台绘制好的12张原始数据图放BWG_Data文件夹下，修改Data_split.py文件所需分割网格代表的像素大小参数并运行便会将原始图片数据按照等份大小进行网状分割，分割后的文件保存在split文件夹内，同一种图片数据会存在同名子文件夹内。
运行Data_process.py文件后程序自动读取不同子文件夹下的同分区图片数据，将图片进行灰度化处理后按照固定顺序叠加在一起并打包成a_b.npy数据包（代表a行b列区域的文件），文件会被储存在Datas文件夹内。
依据收集到的资料进行样本划分，修改Data_augmentation.py中裁切窗口的像素尺寸参数和窗口滑动步长参数。运行程序后会依次读取正负样本数据，对a_b.npy数据文件逐层进行分割法数据增强并重新打包，增强后生成的文件储存在Datas文件夹的New Data子文件夹。
move.py文件运行后会将New Data中的正负样本随机抽取80%划分成训练集，20%划为测试集，最后按照图4.18e的格式存入datasets文件夹内以备后续模型训练。
 **模型训练** 
   准备好数据集后运行annotation.py生成训练所需的bwg_train.txt和bwg_test.txt文本，此文本记录的是训练样本与测试样本的储存路径和对应分类标签。运行train.py文件开始模型训练，训练日志自动保存在logs文件夹中。
 **模型评价** 
    运行eval文件夹下的evaluation.py即可，生成的评价信息将储存在eval_results文件夹内。