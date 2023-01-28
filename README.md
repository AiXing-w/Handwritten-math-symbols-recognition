# Handwritten-math-symbols-recognition
使用常用的CNN神经网络实现数据超过30万条的手写数学符号识别
## 数据集
所用的数据集是来自`kaggle`的`Handwritten math symbols dataset`，其中包括超过30w张图片，共有82个类别。解压后的数据放到extracted_images中作为数据

数据集下载地址：
[Handwritten math symbols dataset](https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols/discussion)

## utils模块
utils中包括数据的加载，模型以及画图展示
### dataLoader
**idxPrepare**
传入数据所在的路径，获取标签与索引的对应关系并以字典的形式保存，并返回由(图片，类别)组成的列表

**image2txt**
传入由(图片，类别)组成的列表，将数据集划分成训练数据和测试数据。并将路径以及对应的标签存放到txt文件中


**MyLoader**
使用torchvsion加载图片

**MyDataLoader**
由于数据量稍微有些大，直接使用torch的数据加载有可能会导致OOM，所以构建了一个数据加载的类，这个类并不会将所有的数据一次性全部加载到内存，而是将存有数据路径和对应的类别先全部加载到内存中，当需要用到数据时，再将数据从磁盘中读到内存当中。

**LoadDataset**
为了方便加载数据，构建了一个数据加载器，传入数据的路径，数据的批量大小和图片的大小，返回训练数据和测试数据的迭代器以供模型的训练
### models
包含各种常见的CNN分类模型，包括LeNet、alexNet、vgg11、NiN、GoogLeNet、resNet18、denseNet模型的构建
**select_model**
用于选择模型，传入模型的名称以及模型的参数，返回选择的分类模型
### plotShow
传入由训练损失、训练精度、测试损失、测试精度组成的字典，然后绘制出图像

## train模块
用于训练模型
**accuracy**
传入预测结果和标签，用于评估预测精度
**train**
传入选择使用的模型，模型参数，训练数据、测试数据、训练轮次、学习率、训练设备、提前终止训练的阈值以及是否保存检查点，进行模型的训练，最终训练结果将会保存在`model_weights`文件夹中
## predict模块
根据不同的参数设置可以进行预测，可选的预测模式有单张图片预测，从文件夹中预测以及随机选取测试集中的数据进行预测
