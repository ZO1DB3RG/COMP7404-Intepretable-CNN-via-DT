import os
import h5py
from tools.load_data import load_data
from tools.init_model import init_model
from tools.train_model import train_model
from tools.showresult import showresult
from tools.lib import *

def compute_metric(root_path,args):
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
    max_sta = showresult(max_epoch,task_model_dataset_labelname_taskid_path, task_model_dataset_labelname_path, root_path, args)

    print("\n")
    print("max_sta = {:.4f}".format(max_sta))







    
    
    
    




