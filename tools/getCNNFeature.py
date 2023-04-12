import os.path

import torch
import torch.nn.functional as F
from tools.get_ilsvrimdb import getI

def hook_fn(module, grad_in, grad_out):
    module.grads = grad_out[0]

def get_x_g(net, im, label, Iter, density,model):
    g = None
    if model == "vgg_vd_16":
        features = net.conv5
        handle = features.register_backward_hook(hook_fn)
        out = net(im, label, Iter, density)
        predicted_cls = torch.max(out, dim=1)[1]
        out[:, predicted_cls].backward()
        g = torch.mean(features.grads, axis=(2, 3), keepdims=True).cpu()
        handle.remove()
        net.zero_grad()

        x = net.conv1(im)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool1(x)
        x = net.conv2(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool2(x)
        x = net.conv3(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool3(x)
        x = net.conv4(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool4(x)
        x = net.conv5(x)
        net.zero_grad()
        y = net(im, label, Iter, density)
        x = x.cpu()
        y = y.cpu()
        # f_map = x.detach()
    return x, g, y

def get_x(net, im, label, Iter, density,model):
    g = None
    if model == "vgg_vd_16":
        x = net.conv1(im)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool1(x)
        x = net.conv2(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool2(x)
        x = net.conv3(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool3(x)
        x = net.conv4(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool4(x)
        x = net.conv5(x)
        x1 = x.detach().cpu().numpy()
        x = net.mask1[0](x, label, Iter, density)
        x2 = x.detach().cpu().numpy()
        x = net.maxpool5[0](x)
        x = x.cpu().clone().data.numpy()
    elif model == "alexnet":
        x = net.conv1(im)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool1(x)

        x = net.conv2(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool2(x)

        x = net.conv3(x)
        x = net.mask1[0](x, label, Iter, density)
        x = net.maxpool3(x)
    elif model == "vgg_s":
        x = net.conv1(im)
        x = F.pad(x, (0, 2, 0, 2))
        x = net.maxpool1(x)

        x = net.conv2(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool2(x)

        x = net.conv3(x)
        x = net.mask1[0](x, label, Iter, density)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool3(x)
    elif model == "vgg_m":
        x = net.conv1(im)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool1(x)

        x = net.conv2(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool2(x)

        x = net.conv3(x)
        x = net.mask1[0](x, label, Iter, density)
        x = net.maxpool3(x)
    elif model == "resnet_18":
        x = net.pad2d_3(im)  # new padding
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.pad2d_1(x)
        x = net.maxpool(x)

        x = net.layer1(x)
        x = net.layer2(x)
        x = net.layer3(x)
        x = net.mask1[0](x, label, Iter, density)
        # f_map = x.detach()

    elif model == "resnet_50":
        x = net.pad2d_3(im)  # new padding
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.pad2d_1(x)
        x = net.maxpool(x)

        x = net.layer1(x)
        x = net.layer2(x)
        x = net.layer3(x)
        x = net.mask1[0](x, label, Iter, density)
        # f_map = x.detach()
    return x, g

def getCNNFeature(dataset_path, obj, net, isFlip, dataMean,epochnum, model):

    if "ilsvrcanimalpart" in dataset_path:
        I = getI(obj, (224,224), isFlip)
    elif "vocpart" in dataset_path:
        I = getI(obj, (224,224), isFlip)
    elif "cub200" in dataset_path:
        I = getI(obj, (224,224), isFlip)

    im = I[0] - dataMean
    im = torch.from_numpy(im).float()
    im = im.unsqueeze(3)
    im = im.permute(3, 2, 0, 1)
    label = torch.ones((1, 1, 1, 1))
    im = im.cuda()
    label = label.cuda()
    net = net.cuda()
    Iter = torch.Tensor([epochnum])
    density = torch.Tensor([0])
    x,g = get_x(net, im, label, Iter, density,model) # type numpy
    return x, I[0], g

def getCNNFeature_gradient(dataset_path, obj, net, isFlip, dataMean,epochnum, model):

    if "ilsvrcanimalpart" in dataset_path:
        I = getI(obj, (224,224), isFlip)
    elif "vocpart" in dataset_path:
        I = getI(obj, (224,224), isFlip)
    elif "cub200" in dataset_path:
        I = getI(obj, (224,224), isFlip)

    im = I[0] - dataMean
    im = torch.from_numpy(im).float()
    im = im.unsqueeze(3)
    im = im.permute(3, 2, 0, 1)
    label = torch.ones((1, 1, 1, 1))
    im = im.cuda()
    label = label.cuda()
    net = net.cuda()
    Iter = torch.Tensor([epochnum])
    density = torch.Tensor([0])
    x, g, y = get_x_g(net, im, label, Iter, density,model) # type numpy
    sd = torch.mean(x, axis=(1,2,3), keepdim=False).squeeze()
    x = torch.sum(x, axis=(2, 3), keepdims=True) / sd
    g = sd * g
    x = x.squeeze()
    y = y.squeeze()
    g = g.squeeze()
    # y_test = torch.matmul(g, x)
    pt_name = obj['filename'].split('.')[0].split('/')[-1] + '.pt'
    torch.save({'x': x, 'g': g, 'y': y, 'sd': sd}, os.path.join('/data/LZL/ICNN/output/cat/x_g_y_sd', pt_name))
    return x, I[0], g


