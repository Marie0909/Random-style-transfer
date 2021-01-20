import numpy as np
import nibabel as nib
import torch
from torch import nn
from scipy import stats
from function import F_compute_sdm


np.seterr(divide='ignore', invalid='ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

height = 144
depth = 144

def F_read_excel(file):
    import xlrd
    numpyindex_list = []
    wb = xlrd.open_workbook(filename=file)
    sheet1 = wb.sheet_by_index(1)
    for i in range(sheet1.nrows-1):
        rows = sheet1.row_values(i+1)
        numpyindex_list.extend([rows])

    return numpyindex_list

def F_get_image_center(image):
    numpyimage = image.squeeze()
    center_coord = numpyimage.shape[0]//2, numpyimage.shape[1]//2

    return center_coord

def F_get_foreground_center(label):
    numpylabel = label.squeeze()
    center_numpylabel = numpylabel[:, :, int(numpylabel.shape[2]/2)]
    center_coord = np.floor(np.mean(np.stack(np.where(center_numpylabel > 0)), -1))

    return center_coord

def F_nifity_imageCrop(numpyimage, center_coord):
    center_x, center_y = center_coord
    shape = numpyimage.shape
    numpyimagecrop = np.zeros((height, depth, shape[2]), dtype=np.float32)

    if (shape[0]<height) :
        numpyimage = np.pad(numpyimage, ((0, height-shape[0]), (0, 0),(0, 0)), 'constant')
    if (shape[1]<depth):
        numpyimage = np.pad(numpyimage, ((0, 0), (0, depth-shape[1]),(0, 0)), 'constant')
   
    if (center_x>=height/ 2) and (center_y>=depth/ 2):
        numpyimagecrop[0:height, 0:depth, :] = numpyimage[int(center_x - height/ 2):int(center_x + height/ 2),
            int(center_y - depth/ 2):int(center_y + depth / 2), :]
    elif (center_x>=height/ 2) and (center_y<depth/ 2):
        numpyimagecrop[0:height, 0:depth, :] = numpyimage[int(center_x - height/ 2):int(center_x + height/ 2), 0:depth, :]      
    elif (center_x<height/ 2) and (center_y>=depth/ 2):
        numpyimagecrop[0:height, 0:depth, :] = numpyimage[0:height, int(center_y - depth/ 2):int(center_y + depth / 2), :]
    else:
        numpyimagecrop[0:height, 0:depth, :] = numpyimage[0:height, 0:depth, :]

    return numpyimagecrop

def LoadTrainingDataset(imagenames, labelnames, ED_ID, ES_ID):
    numpy2Dimage = []
    numpy2Dlabel = []
    numpy2Ddist = []
    numpy2Dslice = []
    NumSlice = 0
    print('loading training image and label: ' + imagenames)

    nibimage = nib.load(imagenames)
    imagedata = nibimage.get_data()
    numpyimage = np.array(imagedata).squeeze()

    niblabel = nib.load(labelnames)
    labeldata = niblabel.get_data()
    numpylabel = np.array(labeldata).squeeze()

    #-------process ED image------
    numpyimage_ED = numpyimage[:, :, :, ED_ID]
    numpylabel_ED = numpylabel[:, :, :, ED_ID]
   
    crop_center_coord_ED = F_get_foreground_center(numpylabel_ED)
    numpyimagecrop_ED = F_nifity_imageCrop(numpyimage_ED, crop_center_coord_ED)  # crop image
    numpylabelcrop_ED = F_nifity_imageCrop(numpylabel_ED, crop_center_coord_ED)  # crop label

    # Convert the 3D image into 2D
    size = numpyimagecrop_ED.shape
    for sliceid in range(size[2]):
        numpy2Dimage.append(np.nan_to_num(stats.zscore(numpyimagecrop_ED[:, :, sliceid])))  # z-score normalization
        numpy2Dlabel.append(numpylabelcrop_ED[:, :, sliceid])
        numpy2Dslice.append(np.array([(sliceid-size[2]/2)/size[2]]))
        NumSlice = NumSlice + 1

        # numpylabel_crop_new = (np.expand_dims(np.expand_dims(numpylabelcrop_ED[:, :, sliceid], 0), 0)>2)*1
        # gt_dis = F_compute_sdm(numpylabel_crop_new, numpylabel_crop_new.shape)
        # numpydist_ED = np.squeeze(np.squeeze(gt_dis, axis=0), axis=0)
        # numpy2Ddist.append(numpydist_ED)
        

    #-------process ES image------
    numpyimage_ES = numpyimage[:, :, :, ES_ID]
    numpylabel_ES = numpylabel[:, :, :, ES_ID]

    crop_center_coord_ES = F_get_foreground_center(numpylabel_ES)
    numpyimagecrop_ES = F_nifity_imageCrop(numpyimage_ES, crop_center_coord_ES)  # crop image
    numpylabelcrop_ES = F_nifity_imageCrop(numpylabel_ES, crop_center_coord_ES)  # crop label


    # Convert the 3D image into 2D
    size = numpyimagecrop_ES.shape
    for sliceid in range(size[2]):
        numpy2Dimage.append(np.nan_to_num(stats.zscore(numpyimagecrop_ES[:, :, sliceid])))  # z-score normalization
        numpy2Dlabel.append(numpylabelcrop_ES[:, :, sliceid])
        numpy2Dslice.append(np.array([(sliceid-size[2]/2)/size[2]]))
        NumSlice = NumSlice + 1

        # numpylabel_crop_new = (np.expand_dims(np.expand_dims(numpylabelcrop_ES[:, :, sliceid], 0), 0)>2)*1
        # gt_dis = F_compute_sdm(numpylabel_crop_new, numpylabel_crop_new.shape)
        # numpydist_ES = np.squeeze(np.squeeze(gt_dis, axis=0), axis=0)
        # numpy2Ddist.append(numpydist_ES)
        
    return numpy2Dimage, numpy2Dlabel, numpy2Dslice, NumSlice

def LoadImageDataset(imagenames, ED_ID, ES_ID):
    numpy2Dimage = []
    NumSlice = 0

    nibimage = nib.load(imagenames)
    imagedata = nibimage.get_data()
    numpyimage = np.array(imagedata).squeeze()

    #-------process ED image------
    numpyimage_ED = numpyimage[:, :, :, ED_ID]  
    crop_center_coord_ED = F_get_image_center(numpyimage_ED)
    numpyimagecrop_ED = F_nifity_imageCrop(numpyimage_ED, crop_center_coord_ED)  # crop image

    # Convert the 3D image into 2D
    size = numpyimagecrop_ED.shape
    for sliceid in range(size[2]):
        numpy2Dimage.append(np.nan_to_num(stats.zscore(numpyimagecrop_ED[:, :, sliceid])))  # z-score normalization
        NumSlice = NumSlice + 1

    #-------process ES image------
    numpyimage_ES = numpyimage[:, :, :, ES_ID]
    crop_center_coord_ES = F_get_image_center(numpyimage_ES)
    numpyimagecrop_ES = F_nifity_imageCrop(numpyimage_ES, crop_center_coord_ES)  # crop image

    # Convert the 3D image into 2D
    size = numpyimagecrop_ES.shape
    for sliceid in range(size[2]):
        numpy2Dimage.append(np.nan_to_num(stats.zscore(numpyimagecrop_ES[:, :, sliceid])))  # z-score normalization
        NumSlice = NumSlice + 1

    return numpy2Dimage, NumSlice, crop_center_coord_ED

def LoadLabelDataset(labelnames, crop_center_coord):
    numpy2Dlabel = []

    niblabel = nib.load(labelnames)
    labeldata = niblabel.get_data()
    numpylabel = np.array(labeldata).squeeze()
    numpylabelcrop = F_nifity_imageCrop(numpylabel, crop_center_coord)  # crop label

    # Convert the 3D image into 2D
    size = numpylabelcrop.shape
    for sliceid in range(size[2]):
        numpy2Dlabel.append(numpylabelcrop[:, :, sliceid])

    return numpy2Dlabel

def ProcessTestDataset(imagename, ED_ID, ES_ID, G_net):

    print('loading test image: ' + imagename)
    nibimage = nib.load(imagename)
    shape = nibimage.shape

    numpyimage, NumSlice, crop_center_coord = LoadImageDataset(imagename, ED_ID, ES_ID)

    for sliceid in range(NumSlice):
        tensorimage = torch.from_numpy(np.array([numpyimage[sliceid]])).unsqueeze(0).float().to(device)
        MIdex_t, MIdex_s = torch.zeros_like(tensorimage), torch.zeros_like(tensorimage)
        MIdex_source, MIdex_target = MIdex_s.repeat(1, 4, 1, 1).to(device), MIdex_t.repeat(1, 4, 1, 1).to(device)
        # output = G_net(tensorimage)       
        # out_Tseg, _ = output
        output = G_net(tensorimage, MIdex_source, MIdex_target)
        out_Tseg, out_Timg, out_Sseg, out_Simg = output

        # from matplotlib import pyplot as plt
        # plt.figure()
        # plt.subplot(231)
        # plt.imshow(out_Tseg[0, 1, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
        # plt.subplot(232)
        # plt.imshow(out_Sseg[0, 1, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
        # plt.subplot(233)
        # plt.imshow(out_Timg[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
        # plt.subplot(234)
        # plt.imshow(out_Simg[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
        # plt.subplot(235)
        # plt.imshow(tensorimage[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
        # plt.show()
        # plt.savefig('output_img.jpg')


        output = np.squeeze(out_Tseg.cpu().detach().numpy(), axis=0)
        outlabel = np.argmax(output, axis=0)
        outputimg = (outlabel == 1) * 1 + (outlabel == 2) * 2 + (outlabel == 3) * 3

        outputimg = outputimg[:, :, np.newaxis]
        if sliceid == 0:
            label = outputimg
        else:
            label = np.concatenate((label, outputimg), axis=-1)

    center_x, center_y = crop_center_coord
    pad_width = ((int(center_x - height//2),int(shape[0] - center_x - height//2)),(int(center_y - depth//2),int(shape[1] - center_y - depth//2)), (0,0))
    predictnumpylabel = np.pad(label, pad_width, 'constant')
    predictnumpylabel_ED = predictnumpylabel[:, :, 0:NumSlice//2]
    predictnumpylabel_ES = predictnumpylabel[:, :, NumSlice//2:NumSlice]

    predictnumpylabel = np.zeros((shape[0], shape[1], shape[2], shape[3]))
    predictnumpylabel[:, :, :, ED_ID] = predictnumpylabel_ED
    predictnumpylabel[:, :, :, ES_ID] = predictnumpylabel_ES
    predictlabel = nib.Nifti1Image(predictnumpylabel, nibimage.affine, nibimage.header)

    predictlabel_ED = nib.Nifti1Image(predictnumpylabel_ED, nibimage.affine, nibimage.header)
    predictlabel_ES = nib.Nifti1Image(predictnumpylabel_ES, nibimage.affine, nibimage.header)

    return predictlabel, predictlabel_ED, predictlabel_ES

def LoadImageDataset_upload(imagename_ED, imagename_ES):
    numpy2Dimage = []
    NumSlice = 0

    nibimage = nib.load(imagename_ED)
    imagedata = nibimage.get_data()
    numpyimage_ED = np.array(imagedata).squeeze()

    nibimage = nib.load(imagename_ES)
    imagedata = nibimage.get_data()
    numpyimage_ES = np.array(imagedata).squeeze()

    #-------process ED image------ 
    crop_center_coord_ED = F_get_image_center(numpyimage_ED)
    numpyimagecrop_ED = F_nifity_imageCrop(numpyimage_ED, crop_center_coord_ED)  # crop image

    # Convert the 3D image into 2D
    size = numpyimagecrop_ED.shape
    for sliceid in range(size[2]):
        numpy2Dimage.append(np.nan_to_num(stats.zscore(numpyimagecrop_ED[:, :, sliceid])))  # z-score normalization
        NumSlice = NumSlice + 1

    #-------process ES image------
    crop_center_coord_ES = F_get_image_center(numpyimage_ES)
    numpyimagecrop_ES = F_nifity_imageCrop(numpyimage_ES, crop_center_coord_ES)  # crop image

    # Convert the 3D image into 2D
    size = numpyimagecrop_ES.shape
    for sliceid in range(size[2]):
        numpy2Dimage.append(np.nan_to_num(stats.zscore(numpyimagecrop_ES[:, :, sliceid])))  # z-score normalization
        NumSlice = NumSlice + 1

    return numpy2Dimage, NumSlice, crop_center_coord_ED

def ProcessTestDataset_upload(imagename_ED, imagename_ES, G_net):

    print('loading test image: ' + imagename_ED)
    nibimage = nib.load(imagename_ED)
    shape = nibimage.shape

    numpyimage, NumSlice, crop_center_coord = LoadImageDataset_upload(imagename_ED, imagename_ES)

    for sliceid in range(NumSlice):
        tensorimage = torch.from_numpy(np.array([numpyimage[sliceid]])).unsqueeze(0).float().to(device)
        MIdex_t, MIdex_s = torch.zeros_like(tensorimage), torch.zeros_like(tensorimage)
        MIdex_source, MIdex_target = MIdex_s.repeat(1, 4, 1, 1).to(device), MIdex_t.repeat(1, 4, 1, 1).to(device)
        # output = G_net(tensorimage)       
        # out_Tseg, _ = output
        tensorlabel = torch.cat([tensorimage, tensorimage, tensorimage, tensorimage], dim=1).float()
        output = G_net(tensorimage, tensorlabel, MIdex_source, MIdex_target)
        out_Tseg, out_Timg, out_Sseg, out_Simg, _, _, _, _, _ = output

        output = np.squeeze(out_Tseg.cpu().detach().numpy(), axis=0)
        outlabel = np.argmax(output, axis=0)
        outputimg = (outlabel == 1) * 3 + (outlabel == 2) * 2 + (outlabel == 3) * 1
        # outputimg = (outlabel == 1) * 1 + (outlabel == 2) * 2 + (outlabel == 3) * 3

        outputimg = outputimg[:, :, np.newaxis]
        if sliceid == 0:
            label = outputimg
        else:
            label = np.concatenate((label, outputimg), axis=-1)

    center_x, center_y = crop_center_coord
    pad_width = ((int(center_x - height//2),int(shape[0] - center_x - height//2)),(int(center_y - depth//2),int(shape[1] - center_y - depth//2)), (0,0))
    predictnumpylabel = np.pad(label, pad_width, 'constant')
    predictnumpylabel_ED = predictnumpylabel[:, :, 0:NumSlice//2]
    predictnumpylabel_ES = predictnumpylabel[:, :, NumSlice//2:NumSlice]

    predictlabel_ED = nib.Nifti1Image(predictnumpylabel_ED, nibimage.affine, nibimage.header)
    predictlabel_ES = nib.Nifti1Image(predictnumpylabel_ES, nibimage.affine, nibimage.header)

    return predictlabel_ED, predictlabel_ES
