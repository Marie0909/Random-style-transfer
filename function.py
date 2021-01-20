import numpy as np
from torch import nn
import torch
import collections
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def F_mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   
		os.makedirs(path)           

def F_loss(output, image, label):
    out_seg, out_img = output
    
    lossfunc1 = nn.CrossEntropyLoss().to(device)
    lossfunc2 = nn.L1Loss().to(device)
    label = label.squeeze(1)
    L_seg = lossfunc1(out_seg, label)
    L_rec = torch.mean(abs(out_img - image))

    return L_seg, L_rec

def F_loss_G(output, image, label):
    out_Tseg, out_Timg, out_Sseg, out_Simg = output
    
    lossfunc1 = nn.CrossEntropyLoss().to(device)
    lossfunc2 = nn.L1Loss().to(device)
    
    label = label.squeeze(1)
    L_seg = lossfunc1(out_Tseg, label)  
    L_shape = lossfunc1(out_Sseg, label) 
    L_rec = torch.mean(abs(out_Simg - image))

    from matplotlib import pyplot as plt
    plt.figure()
    output = out_Tseg.cpu().detach().numpy()
    outlabel = np.argmax(output, axis=1)
    outputimg_T = (outlabel == 1) * 1 + (outlabel == 2) * 2 + (outlabel == 3) * 3   
    output = out_Sseg.cpu().detach().numpy()
    outlabel = np.argmax(output, axis=1)
    outputimg_S = (outlabel == 1) * 1 + (outlabel == 2) * 2 + (outlabel == 3) * 3
    plt.subplot(231)
    plt.imshow(outputimg_T[0, :, :], cmap=plt.cm.gray)
    plt.subplot(232)
    plt.imshow(outputimg_S[0, :, :], cmap=plt.cm.gray)
    plt.subplot(233)
    plt.imshow(out_Timg[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
    plt.subplot(234)
    plt.imshow(out_Simg[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
    plt.subplot(235)
    plt.imshow(image[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
    plt.subplot(236)
    plt.imshow(label[0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
    plt.savefig('output_img.jpg')

    return L_seg, L_shape, L_rec


def F_loss_D(out_real, out_false, MIdex_s, MIdex_t):
    disc_real, cls_real = out_real
    disc_fake, cls_fake = out_false
    # disc_real = out_real
    # disc_fake = out_false
    gen_cost = -torch.mean(disc_fake)
    disc_cost = torch.mean(disc_fake) - torch.mean(disc_real)

    lossfunc1 = nn.BCELoss().to(device)
    L_cls_real = lossfunc1(cls_real, MIdex_s)
    L_cls_fake = lossfunc1(cls_fake, MIdex_t) #torch.mean(-MIdex_t*torch.log(cls_fake))
    return gen_cost, disc_cost, L_cls_real, L_cls_fake

def F_gradient_penalty(D_net, real_data, generated_data):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data).to(device)
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True).to(device)

    # Calculate probability of interpolated examples
    prob_interpolated,_ = D_net(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                            create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)
    #self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return ((gradients_norm - 1) ** 2).mean()

def F_compute_sdm(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdm(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdm to [-1,1]----optional
    """
    T = 50 #to constrain the value of sdm ----optional
    img_gt = img_gt.astype(np.uint8)
    normalized_sdm = T*np.ones(out_shape) #np.zeros(out_shape)
    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                #sdm = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                sdm = negdis - posdis
                sdm[boundary==1] = 0
                normalized_sdm[b][c] = sdm
                #assert np.min(sdm) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
                #assert np.max(sdm) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return np.clip(normalized_sdm, -T, T)
    
def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def LabelDice(A, B, class_labels):
    '''
    :param A: (n_batch, 1, n_1, ..., n_k)
    :param B: (n_batch, 1, n_1, ..., n_k)
    :param class_labels: list[n_class]
    :return: (n_batch, n_class)
    '''
    return Dice(torch.cat([1 - torch.clamp(torch.abs(A - i), 0, 1) for i in class_labels], 1),
                torch.cat([1 - torch.clamp(torch.abs(B - i), 0, 1) for i in class_labels], 1))

def Dice(A, B):
    '''
    A: (n_batch, n_class, n_1, n_2, ..., n_k)
    B: (n_batch, n_class, n_1, n_2, ..., n_k)
    return: (n_batch, n_class)
    '''
    eps = 1e-8
#    assert torch.sum(A * (1 - A)).abs().item() < eps and torch.sum(B * (1 - B)).abs().item() < eps
    A = A.flatten(2).float(); B = B.flatten(2).float()
    ABsum = A.sum(-1) + B.sum(-1)
    return 2 * torch.sum(A * B, -1) / (ABsum + eps)


#-----------------load net param-----------------------------
def F_LoadsubParam(net_param, sub_net, target_net):
    print(net_param)
    state_dict = torch.load(net_param, map_location='cpu')
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    sub_net.load_state_dict(new_state_dict)

    # ---------------load the param of Seg_net into SSM_net---------------
    sourceDict = sub_net.state_dict()
    targetDict = target_net.state_dict()
    target_net.load_state_dict({k: sourceDict[k] if k in sourceDict else targetDict[k] for k in targetDict})

def F_LoadParam(net_param, target_net):
    print(net_param)
    state_dict = torch.load(net_param, map_location='cpu')
    target_net.load_state_dict(state_dict)

def F_LoadParam_test(net_param, target_net):
    print(net_param)
    state_dict = torch.load(net_param, map_location='cpu')

    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    target_net.load_state_dict(new_state_dict)
