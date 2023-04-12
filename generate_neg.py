import os
import random
import shutil
import xml.etree.ElementTree as ET

datasets_JPEG = '/data/LZL/ICNN/datasets/VOCdevkit/VOC2010/JPEGImages'
datasets_xml = '/data/LZL/ICNN/datasets/VOCdevkit/VOC2010/Annotations'
MaxObjNum = 1000
target_cate = 'cat'

if os.path.exists('./neg') == False:
    os.makedirs('./neg')
    if os.path.exists(datasets_JPEG) == False:
        print('no such directory')
        exit(0)
    else:
        all_neg_image = os.listdir(datasets_JPEG)
        #读取xml文件，将所有不是cat的图像pop
        for i, img in enumerate(all_neg_image):
            xml_name = os.path.splitext(img)[0] + '.xml'
            xml_path = os.path.join(datasets_xml, xml_name)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            cls = root.find('object').find('name').text
            if cls == target_cate:
                all_neg_image.pop(i)
                continue
        print('number of the neg img: {}'.format(len(all_neg_image)))
        print('random sampling: ')
        for i, ind in enumerate(random.sample(range(0, len(all_neg_image)),MaxObjNum)):
            new_name = "%05d.JPEG" % (i+1)
            new_path = os.path.join('./neg', new_name)

            old_path = os.path.join(datasets_JPEG, os.listdir(datasets_JPEG)[ind])
            shutil.copyfile(old_path, new_path)
            if os.path.exists(new_path):
                print("copy complete for pic: {}".format(os.listdir(datasets_JPEG)[ind]))
            else:
                print('fail')
