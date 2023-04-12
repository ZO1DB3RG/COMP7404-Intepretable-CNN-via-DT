# Intepretable-CNN-via-DT
Project code for HKU COMP7404

## Notice
- **model**

The project now supports vgg_vd_16


                                         


- **dataset**

The project now supports **vocpart, ilsvrc animalpart, cub200, 
                         celeba, voc2010_crop, helen**.
（
从网站http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
    下载VOCPART数据集，放到./datasets下）
    
    
正样本集每一幅图像输入网络后的特征图与梯度向量都已保存为.pt文件，文件名为对应的图像名，方便之后重新找到这张图像。
地址：https://drive.google.com/file/d/1pY0D5yWl57wBUQkDQjuQDP5hnpgSDlm1/view?usp=share_link

同时也需要下载一个和VOC相关的ground truth：地址为：https://drive.google.com/drive/folders/17oQ-sPDB5EfZETwCLUBOTnUz8RGB3Uir?usp=share_link
