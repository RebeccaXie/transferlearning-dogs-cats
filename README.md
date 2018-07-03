# transferlearning-dogs-cats
1. transferlearning-incepV3.py 和 lenet_dogs_cats.py为主要程序

2. prepossess.py 是将原Kaggel数据集选取部分数据集；存储结构为：
data/ dogs..cats -- 同级文件夹

3. image-info-process.py 是将各文件名按比例随机分为测试集和训练集，以字典形式存储起来例如{cat:{test,train}}其中{test:1.jpg}，得到image_info.json 文件。后通过文件名索引得到图片地址，作为transferlearning-incepV3.py的输入。
./data 为其中路径输入设置

4. prep_resize.py 将原图片通过缩放形成尺寸为28*28的图片再储存，并以image_info.json 文件内容对上述数据重新排布；存储结构为：
data1/train/dogs..cats。 其中./data1/train 和 ./data1/test 为lenet_dogs_cats.py中路径输入设置

5. inception-v3 文件可从 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz' 下载。

6. 本项目参考了《Tensorflow实战Google深度学习框架》；
利用tensorflow输出预训练模型中间层名称数据https://www.cnblogs.com/edgar-/p/6519245.html
