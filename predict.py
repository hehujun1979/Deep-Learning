'''
1、如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要将预测结果保存成txt，可以利用open打开txt文件，使用write方法写入txt，参考txt_annotation.py文件。
'''
import numpy as np
from classification import Classification

classfication = Classification()

while True:
    data_path = input('Input image filename:')
    try:
        image = np.load(data_path)
    except:
        print('Open Error! Try again!')
        continue
    else:
        class_name = classfication.detect_image(image)
        print(class_name)
