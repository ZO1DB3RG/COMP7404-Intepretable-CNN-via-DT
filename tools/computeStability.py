import os

import cv2
import tqdm
from scipy.io import loadmat
import h5py
import numpy as np
from tools.getDistSqrtVar import getDistSqrtVar
from tools.getCNNFeature import getCNNFeature, getCNNFeature_gradient
from tools.get_ilsvrimdb import readAnnotation as ilsvr_readAnnotation
from tools.get_cubimdb import readAnnotation as cub_readAnnotation
from tools.get_vocimdb import readAnnotation as voc_readAnnotation
import scipy.io as scio

def x2P(idx_h, idx_w, layerID, convnet):
    idx_h = idx_h[np.newaxis, :]
    idx_w = idx_w[np.newaxis, :]
    pHW = np.concatenate((idx_h, idx_w), axis=0)
    Stride = convnet['targetStride'][layerID-1]
    centerStart = convnet['targetCenter'][layerID-1]
    pHW = centerStart + (pHW-1) * Stride
    return pHW


def computeStability(root_path,dataset,dataset_path, truthpart_path, label_name, net, model, convnet, layerID, epochnum, partList, partRate, imdb_mean, selectPatternRatio, patchNumPerPattern):

    if "ilsvrcanimalpart" in dataset_path:
        objset = ilsvr_readAnnotation(dataset_path, label_name)
    elif "vocpart" in dataset_path:
        objset = voc_readAnnotation(root_path, dataset, dataset_path, label_name)
    elif "cub200" in dataset_path:
        objset = cub_readAnnotation(dataset_path, label_name)

    imgNum = len(objset)
    partNum = len(partList)
    validImg = np.zeros(imgNum)
    for i in range(partNum):
        partID = partList[i]
        file_path = os.path.join(truthpart_path,label_name, "truth_part"+str(0) + str(partID)+'.mat')
        truth_center = scio.loadmat(file_path)['truth']['pHW_center']
        for img in range(imgNum):
            if type(truth_center[0][img]) is np.ndarray:
                validImg[img] = True

    patNum = round(512*partRate)
    selectedPatternNum = round(patNum*selectPatternRatio)
    pos = np.zeros((2,patNum,imgNum))
    score = np.zeros((patNum, imgNum))
    isFlip = False
    for imgID in tqdm.tqdm(range(imgNum)):
        if(validImg[imgID]==0):
            continue
        x,I, g = getCNNFeature(dataset_path,objset[imgID],net,isFlip,imdb_mean, epochnum, model) # get after conv_mask feature
        x = x[:,0:patNum,:,:] #选择想要的part通道，此处全选
        x = np.squeeze(x,axis=0)
        xh = x.shape[1]
        v = np.max(x, axis=1)
        idx = np.argmax(x, axis=1)
        tmp = np.argmax(v, axis=1)
        v = np.max(v, axis=1) #每一张特征图的最大值
        idx = idx.reshape(idx.shape[0] * idx.shape[1])
        idx_h = idx[tmp + np.array(range(0, patNum)) * xh]  # idx_h.shape=(patNum,)
        idx_w = tmp  # idx_w.shape=(patNum,)
        theScore = v  # v.shape=(patNum,)
        thePos = x2P(idx_h,idx_w,layerID,convnet)
        pos[:,:,imgID] = thePos
        score[:,imgID] = theScore
    ih = I.shape[0]
    iw = I.shape[1]
    distSqrtVar = getDistSqrtVar(truthpart_path, pos, score, patchNumPerPattern, partList, label_name)
    min_channel = np.argmin(distSqrtVar)
    distSqrtVar = np.sort(distSqrtVar[np.isnan(distSqrtVar) == 0])

    stability = np.mean(distSqrtVar[0:min(selectedPatternNum, len(distSqrtVar))])/np.sqrt(np.power(ih,2)+np.power(iw,2))

    return stability

def get_plot(root_path,dataset,dataset_path, truthpart_path, label_name, net, model, convnet, layerID, epochnum, partList, partRate, imdb_mean, selectPatternRatio, patchNumPerPattern):

    if "ilsvrcanimalpart" in dataset_path:
        objset = ilsvr_readAnnotation(dataset_path, label_name)
    elif "vocpart" in dataset_path:
        objset = voc_readAnnotation(root_path, dataset, dataset_path, label_name)
    elif "cub200" in dataset_path:
        objset = cub_readAnnotation(dataset_path, label_name)

    imgNum = len(objset)
    partNum = len(partList)
    validImg = np.zeros(imgNum)
    for i in range(partNum):
        partID = partList[i]
        file_path = os.path.join(truthpart_path,label_name, "truth_part"+str(0) + str(partID)+'.mat')
        truth_center = scio.loadmat(file_path)['truth']['pHW_center']
        for img in range(imgNum):
            if type(truth_center[0][img]) is np.ndarray:
                validImg[img] = True

    patNum = round(512*partRate)
    selectedPatternNum = round(patNum*selectPatternRatio)
    pos = np.zeros((2,patNum,imgNum))
    score = np.zeros((patNum, imgNum))
    isFlip = False
    for imgID in tqdm.tqdm(range(imgNum)):
        if(validImg[imgID]==0):
            continue
        x,I, g = getCNNFeature(dataset_path,objset[imgID],net,isFlip,imdb_mean, epochnum, model) # get after conv_mask feature
        x = x[:,0:patNum,:,:] #选择想要的part通道，此处全选
        x = np.squeeze(x,axis=0)
        xh = x.shape[1]
        v = np.max(x, axis=1)
        idx = np.argmax(x, axis=1)
        tmp = np.argmax(v, axis=1)
        v = np.max(v, axis=1) #每一张特征图的最大值
        idx = idx.reshape(idx.shape[0] * idx.shape[1])
        idx_h = idx[tmp + np.array(range(0, patNum)) * xh]  # idx_h.shape=(patNum,)
        idx_w = tmp  # idx_w.shape=(patNum,)
        theScore = v  # v.shape=(patNum,)
        thePos = x2P(idx_h,idx_w,layerID,convnet)
        pos[:,:,imgID] = thePos
        score[:,imgID] = theScore
    ih = I.shape[0]
    iw = I.shape[1]
    distSqrtVar = getDistSqrtVar(truthpart_path, pos, score, patchNumPerPattern, partList, label_name)

    min_channel = np.argmin(distSqrtVar)
    chosen_channel = np.argsort(distSqrtVar[:, 0])[0]

    distSqrtVar = np.sort(distSqrtVar[np.isnan(distSqrtVar) == 0])
    stability = np.mean(distSqrtVar[0:min(selectedPatternNum, len(distSqrtVar))])/np.sqrt(np.power(ih,2)+np.power(iw,2))

    if not os.path.exists(os.path.join(root_path, 'output', label_name, 'par_1')):
        os.makedirs(os.path.join(root_path, 'output', label_name, 'par_1'))

    for imgID in tqdm.tqdm(range(imgNum)):
        if (validImg[imgID] == 0):
            continue
        x, _, g = getCNNFeature(dataset_path, objset[imgID], net, isFlip, imdb_mean, epochnum,
                             model)  # get after conv_mask feature
        I = cv2.imread(objset[imgID]['filename'])
        ori_w, ori_h = I.shape[0], I.shape[1]
        I = cv2.resize(I, (224, 224), interpolation=cv2.INTER_LINEAR)
        x = x[:, 0:patNum, :, :]  # 选择想要的part通道，此处全选
        x = np.squeeze(x, axis=0)

        mask = x[chosen_channel].copy()
        mask = cv2.resize(mask, (I.shape[0], I.shape[1]))
        thresh_rate = 0.2
        thresh = thresh_rate * (np.max(mask) - np.min(mask))

        mask[mask < thresh] = 0.0
        mask[mask > thresh] = 1.0

        img_mask = np.multiply(I, mask[:, :, np.newaxis])
        img_mask = I * 0.3 + img_mask

        cv2.imwrite(os.path.join(root_path, 'output', label_name,'par_1', ("%05d.jpg") % (imgID+1)), img_mask)

    return stability

def save_g_x(root_path,dataset,dataset_path, truthpart_path, label_name, net, model, convnet, layerID, epochnum, partList, partRate, imdb_mean, selectPatternRatio, patchNumPerPattern):

    if "ilsvrcanimalpart" in dataset_path:
        objset = ilsvr_readAnnotation(dataset_path, label_name)
    elif "vocpart" in dataset_path:
        objset = voc_readAnnotation(root_path, dataset, dataset_path, label_name)
    elif "cub200" in dataset_path:
        objset = cub_readAnnotation(dataset_path, label_name)

    imgNum = len(objset)
    partNum = len(partList)
    validImg = np.zeros(imgNum)
    for i in range(partNum):
        partID = partList[i]
        file_path = os.path.join(truthpart_path,label_name, "truth_part"+str(0) + str(partID)+'.mat')
        truth_center = scio.loadmat(file_path)['truth']['pHW_center']
        for img in range(imgNum):
            if type(truth_center[0][img]) is np.ndarray:
                validImg[img] = True

    patNum = round(512*partRate)
    selectedPatternNum = round(patNum*selectPatternRatio)
    pos = np.zeros((2,patNum,imgNum))
    score = np.zeros((patNum, imgNum))
    isFlip = False
    for imgID in tqdm.tqdm(range(imgNum)):
        if(validImg[imgID]==0):
            continue
        x, I, g = getCNNFeature_gradient(dataset_path, objset[imgID], net, isFlip, imdb_mean, epochnum,
                                model)  # get after conv_mask feature
    return True