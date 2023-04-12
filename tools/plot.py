import os
import h5py
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
import cv2
from tools.GradCAM import GradCAM
from tools.lib import *
from tools.showresult import show_plot, show_grad


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap) #将cam的结果转成伪彩色图片
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) #使用opencv方法后，得到的一般都是BGR格式，还要转化为RGB格式
        # OpenCV中图像读入的数据格式是numpy的ndarray数据格式。是BGR格式，取值范围是[0,255].
    heatmap = np.float32(heatmap) / 255. #缩放到[0,1]之间

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255*cam)

def plot_for_one_image(root_path,args):
    # task/classification
    task_path = os.path.join(root_path,'task',args.task_name)
    make_dir(task_path)
    # task/classification/vgg_vd_16
    task_model_path = os.path.join(task_path,args.model)
    make_dir(task_model_path)
    task_model_dataset_path = os.path.join(task_model_path,args.dataset)
    make_dir(task_model_dataset_path)
    if args.dataset!='helen' and args.dataset!='celeba' and args.dataset!='cubsample':
        task_model_dataset_labelname_path = os.path.join(task_model_dataset_path,args.label_name)
        make_dir(task_model_dataset_labelname_path)
    else:
        task_model_dataset_labelname_path = task_model_dataset_path
    task_model_dataset_labelname_taskid_path = os.path.join(task_model_dataset_labelname_path,str(args.task_id))
    make_dir(task_model_dataset_labelname_taskid_path)

    net_path = args.net_path
    net = load_model(net_path)

    net_path = args.net_path
    max_epoch = int(os.path.split(net_path)[1].split('.')[0].split('-')[1])

    # calculate stability
    max_sta = show_plot(max_epoch, task_model_dataset_labelname_taskid_path, task_model_dataset_labelname_path,root_path, args)

    print("\n")
    print("max_sta = {:.4f}".format(max_sta))

def grad_for_one_image(root_path,args):
    # task/classification
    task_path = os.path.join(root_path,'task',args.task_name)
    make_dir(task_path)
    # task/classification/vgg_vd_16
    task_model_path = os.path.join(task_path,args.model)
    make_dir(task_model_path)
    task_model_dataset_path = os.path.join(task_model_path,args.dataset)
    make_dir(task_model_dataset_path)
    if args.dataset!='helen' and args.dataset!='celeba' and args.dataset!='cubsample':
        task_model_dataset_labelname_path = os.path.join(task_model_dataset_path,args.label_name)
        make_dir(task_model_dataset_labelname_path)
    else:
        task_model_dataset_labelname_path = task_model_dataset_path
    task_model_dataset_labelname_taskid_path = os.path.join(task_model_dataset_labelname_path,str(args.task_id))
    make_dir(task_model_dataset_labelname_taskid_path)
    net_path = args.net_path
    max_epoch = int(os.path.split(net_path)[1].split('.')[0].split('-')[1])

    # calculate stability
    max_sta = show_grad(max_epoch, task_model_dataset_labelname_taskid_path, task_model_dataset_labelname_path,root_path, args)

    print("\n")
    if max_sta:
        print("Success!")




    
    
    
    




