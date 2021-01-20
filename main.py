import numpy as np
import torch
from torch import nn
import nibabel as nib
from torch.utils.data import DataLoader
import torch.utils.data as data
from torch import optim
import os
import time
import glob
from scipy import stats
from torch.nn import DataParallel
from torch.backends import cudnn
import collections
import sys

from loaddata import LoadTrainingDataset, F_get_foreground_center, ProcessTestDataset, LoadImageDataset, F_get_image_center, F_read_excel, ProcessTestDataset_upload
from network import Generator, Discriminator, SegNet_2task
from function import F_LoadParam_test, F_LoadParam, F_loss_G, F_loss_D, F_loss, F_gradient_penalty, F_mkdir

#Root_DIR = '/home/zibin112/Li_Lei/MyoPS2020/'
Root_DIR = '/home/lilei/MM_Challenge/'

TRAIN_DIR_PATH = Root_DIR + 'train_data/'
TEST_DIR_PATH = Root_DIR + 'test_data/'
TRAIN_SAVE_DIR_Seg = Root_DIR + 'Script_ori/result_model/'
Seglossfile = Root_DIR + 'Script_ori/lossfile/L_seg.txt'
imglossfile1 = Root_DIR + 'Script_ori/lossfile/L_rec.txt'
imglossfile2 = Root_DIR + 'Script_ori/lossfile/L_shape.txt'
imglossfile3 = Root_DIR + 'Script_ori/lossfile/L_adv_G.txt'
imglossfile4 = Root_DIR + 'Script_ori/lossfile/L_adv_D.txt'
imglossfile5 = Root_DIR + 'Script_ori/lossfile/L_cls_r.txt'
imglossfile6 = Root_DIR + 'Script_ori/lossfile/L_cls_f.txt'

WORKERSNUM = 16 #16
BatchSize = 30
NumEPOCH = 250

LEARNING_RATE = 3e-4#3e-4
REGULAR_RATE = 0.95
WEIGHT_DECAY = 1e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrainingDataset(data.Dataset):
    def __init__(self, datapath):
        self.datafile = glob.glob(datapath + '*/*')#A1_60_B2_40_Labeled
        self.NumSlice = 0
        self.numpyimage = []
        self.numpylabel = []
        self.numpydist = []
        self.numpyMIndex = []
        self.numpy2Dsliceindex = []

        self.numpyindex_list = F_read_excel(datapath + '/M&Ms Dataset Information.xlsx')
        for subjectid in range(len(self.datafile)):
            # if subjectid > 2:
            #     break
            imagename = self.datafile[subjectid] + '/image.nii.gz'
            labelname = self.datafile[subjectid] + '/label.nii.gz'

            subjectname = self.datafile[subjectid][-6:]
            datainfo =[s for s in self.numpyindex_list if subjectname in s]
            Vendor_ID, Centre_ID, ED_ID, ES_ID = datainfo[0][1], int(datainfo[0][2]), int(datainfo[0][3]), int(datainfo[0][4])
            
            numpy2Dimage, numpy2Dlabel, numpy2Dsliceindex, NumSlice = LoadTrainingDataset(imagename, labelname, ED_ID, ES_ID)
            
            if Vendor_ID == 'A':
                numpyMIndex = np.array([1, 0, 0, 0])
            elif Vendor_ID == 'B':
                numpyMIndex = np.array([0, 1, 0, 0])
            elif Vendor_ID == 'C':
                numpyMIndex = np.array([0, 0, 1, 0])
            elif Vendor_ID == 'D':
                numpyMIndex = np.array([0, 0, 0, 1])
            numpyMIndex_new = [[numpyMIndex] for i in range(NumSlice)]              
            self.numpyMIndex.extend(numpyMIndex_new)

            self.numpyimage.extend(numpy2Dimage)
            self.numpylabel.extend(numpy2Dlabel)
            self.numpy2Dsliceindex.extend(numpy2Dsliceindex)
            self.NumSlice = self.NumSlice + NumSlice
            

    def __getitem__(self, item):

        numpyimage = np.array([self.numpyimage[item]])
        tensorimage = torch.from_numpy(numpyimage).float()

        numpylabel = np.array([self.numpylabel[item]])
        tensorlabel = torch.from_numpy(numpylabel).long()

        numpysliceindex = np.array([self.numpy2Dsliceindex[item]])
        tensorsliceID = torch.from_numpy(numpysliceindex).float()

        numpyMIndex_source = np.array(self.numpyMIndex[item])
        tensorMIndex_s = torch.from_numpy(numpyMIndex_source).float().squeeze()

        numpyMIndex_target = np.eye(4, dtype=int)[np.random.randint(0, 4-1)] #random code for target domain
        tensorMIndex_t = torch.from_numpy(numpyMIndex_target).float()

        return tensorimage, tensorlabel, tensorsliceID, tensorMIndex_s.unsqueeze(-1).unsqueeze(-1), tensorMIndex_t.unsqueeze(-1).unsqueeze(-1)

    def __len__(self):
        return self.NumSlice

def Train_Validate_GAN(dataload, G_net, D_net, epoch, G_optimizer, D_optimizer, savedir):
    start_time = time.time()
    flearning_rate = LEARNING_RATE*(REGULAR_RATE**(epoch//50))
    fregular_rate = 1.0
    f = open(Seglossfile, 'a')
    f1 = open(imglossfile1, 'a')
    f2 = open(imglossfile2, 'a')
    f3 = open(imglossfile3, 'a')
    f4 = open(imglossfile4, 'a')
    f5 = open(imglossfile5, 'a')
    f6 = open(imglossfile6, 'a')

    for i, (image, label, sliceID, MIdex_s, MIdex_t) in enumerate(dataload):
        image, label, sliceID, MIdex_s, MIdex_t = image.to(device), label.to(device), sliceID.to(device), MIdex_s.to(device), MIdex_t.to(device)
        n_batch, nx, ny = image.shape[0], image.shape[2], image.shape[3]
        MIdex_source, MIdex_target = MIdex_s.repeat(1, 1, nx, ny), MIdex_t.repeat(1, 1, nx, ny)
        for param_group in G_optimizer.param_groups:
            param_group['lr'] = flearning_rate
        G_net.train()        
        G_optimizer.zero_grad()
        label_new = torch.cat([(label == 0)* 1.0, (label == 1)* 1.0, (label == 2)* 1.0, (label == 3) * 1.0], dim=1).float()
        G_output = G_net(image, label_new, MIdex_target, MIdex_source)
        _, out_Timg, _, _ = G_output
        
        L_seg, L_shape, L_rec = F_loss_G(G_output, image, label)

        for param_group in D_optimizer.param_groups:
            param_group['lr'] = flearning_rate
        D_net.train()
        D_optimizer.zero_grad()
        D_output_r = D_net(image) #what's real image?
        D_output_f = D_net(out_Timg)

        lambda_gp = 10
        lambda_cls = 100 #10
        lambda_adv = 1

        lambda_rec = 100 #100
        lambda_seg = 100 #100
        lambda_shape = epoch//30

        L_adv_G, disc_cost, L_cls_r, L_cls_f = F_loss_D(D_output_r, D_output_f, MIdex_s, MIdex_t)
        gradient_penalty = F_gradient_penalty(D_net, image, out_Timg)       
        L_adv_D = disc_cost + lambda_gp*gradient_penalty       
        D_loss = lambda_adv*L_adv_D + lambda_cls*L_cls_r        
        G_loss = lambda_adv*L_adv_G + lambda_rec*L_rec + lambda_seg*(L_seg) + lambda_shape*(L_shape) + lambda_cls*L_cls_f
        G_loss.backward(retain_graph=True)
        G_optimizer.step()

        D_loss.backward(retain_graph=True)
        D_optimizer.step()

        f.write(str(L_seg.item()))
        f.write('\n')
        f1.write(str(L_rec.item()))
        f1.write('\n')
        f2.write(str(L_shape.item()))
        f2.write('\n')
        f3.write(str(L_adv_G.item()))
        f3.write('\n')
        f4.write(str(L_adv_D.item()))
        f4.write('\n')
        
        f5.write(str(L_cls_r.item()))
        f5.write('\n')
        f6.write(str(L_cls_f.item()))
        f6.write('\n')

        if i % 50 == 0:
            if i > 1:
                flearning_rate = flearning_rate * fregular_rate
            print('epoch %d , %d th, G-Net learning_rate:%.10f , loss: %.10f' % (epoch, i, flearning_rate, G_loss.item()))
            print('epoch %d , %d th, D-Net learning_rate:%.10f , loss: %.10f' % (epoch, i, flearning_rate, D_loss.item()))

    print('epoch %d , %d th, G-Net learning_rate:%.10f , loss: %.10f' % (epoch, i, flearning_rate, G_loss.item()))
    print('epoch %d , %d th, D-Net learning_rate:%.10f , loss: %.10f' % (epoch, i, flearning_rate, D_loss.item()))
    strNetSaveName = 'G_net_with_%d.pkl' % epoch
    torch.save(G_net.state_dict(), os.path.join(savedir, strNetSaveName))
    strNetSaveName = 'D_net_with_%d.pkl' % epoch
    torch.save(D_net.state_dict(), os.path.join(savedir, strNetSaveName))
    end_time = time.time()
    print('---------------- Train Seg-Net: ' + strNetSaveName + ' , epoch %d cost time : %3.2f ----------------' % (epoch, end_time - start_time))

def main():
    is_for_training = True

    # if len(sys.argv) > 2:
    #     INPUT_DIR = int(sys.argv[1]) #input_data_directory
    #     OUTPUT_DIR = int(sys.argv[2]) #output_data_directory
    #     F_mkdir(OUTPUT_DIR)

    if is_for_training:
        G_net = Generator(1+4, 4).to(device)
        D_net = Discriminator().to(device)
        dataset = TrainingDataset(TRAIN_DIR_PATH)
        data_loader = DataLoader(dataset, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM, pin_memory=True)

        cudnn.benchmark = True #False
        G_net, D_net = DataParallel(G_net,device_ids=[0, 1]), DataParallel(D_net, device_ids=[0, 1])
        G_optimizer = optim.Adam(G_net.parameters())
        D_optimizer = optim.Adam(D_net.parameters())

        # Seg_net_param = TRAIN_SAVE_DIR_Seg + 'G_net_with_150.pkl'
        # F_LoadParam(Seg_net_param, G_net)
        # Seg_net_param = TRAIN_SAVE_DIR_Seg + 'D_net_with_150.pkl'
        # F_LoadParam(Seg_net_param, D_net)

        # for parameter in D_net.D_cls.parameters():
        #     parameter.requires_grad = False 
        # G_optimizer = optim.Adam(filter(lambda p: p.requires_grad, G_net.parameters()))
        # D_optimizer = optim.Adam(filter(lambda p: p.requires_grad, D_net.parameters()))

        for epoch in range(NumEPOCH):
            # epoch = epoch + 151     
            # if (epoch <= 50) or (epoch > 100 and epoch <= 150) or (epoch > 200 and epoch <= 300):
            #     for parameter in G_net.parameters():
            #         parameter.requires_grad = False
            # elif (epoch > 50 and epoch <= 100) or (epoch > 150 and epoch <= 200):
            #     for parameter in D_net.parameters():
            #         parameter.requires_grad = False          
            Train_Validate_GAN(data_loader, G_net, D_net, epoch, G_optimizer, D_optimizer, TRAIN_SAVE_DIR_Seg)
    else:
        str_for_action = 'testing'
        print(str_for_action + ' .... ')
        G_net = Generator(1+4, 4).to(device)
        # Seg_net_param = '/segmentation_model/G_net_with_935.pkl'
        Seg_net_param = TRAIN_SAVE_DIR_Seg + 'G_net_with_249.pkl'
        F_LoadParam(Seg_net_param, G_net)
        G_net.eval()

        datafile = glob.glob(Root_DIR + '/ACDC_100/*')

        
        #datafile = glob.glob(INPUT_DIR + '/mnms/*/')
        #datafile = glob.glob(TEST_DIR_PATH + '*/*')
        #datafile = glob.glob(TRAIN_DIR_PATH + '/Unlabeled/*') #C4_Unlabeled
        #numpyindex_list = F_read_excel(TRAIN_DIR_PATH + '/M&Ms Dataset Information.xlsx')
               
        for subjectid in range(len(datafile)):
            # subjectname = datafile[subjectid][-7:-1]
            # imagename_ED = datafile[subjectid] + subjectname + '_sa_ED.nii.gz'
            # imagename_ES = datafile[subjectid] + subjectname + '_sa_ES.nii.gz'                       
            # predictlabel_ED, predictlabel_ES = ProcessTestDataset_upload(imagename_ED, imagename_ES, G_net)
            # savefold = os.path.join(OUTPUT_DIR + '/' + subjectname + '_sa_ED.nii.gz')           
            # nib.save(predictlabel_ED, savefold)
            # savefold = os.path.join(OUTPUT_DIR + '/' + subjectname + '_sa_ES.nii.gz')
            # nib.save(predictlabel_ES, savefold)

            # imagename = datafile[subjectid] + '/image.nii.gz'
            # subjectname = datafile[subjectid][-6:]
            # datainfo =[s for s in numpyindex_list if subjectname in s]
            # Vendor_ID, Centre_ID, ED_ID, ES_ID = datainfo[0][1], int(datainfo[0][2]), int(datainfo[0][3]), int(datainfo[0][4])     
            # predictlabel, predictlabel_ED, predictlabel_ES = ProcessTestDataset(imagename, ED_ID, ES_ID, G_net)
            # savefold = os.path.join(datafile[subjectid] + '/label_predict_372.nii.gz') #label_predict
            # nib.save(predictlabel, savefold)
            # # savefold = os.path.join(datafile[subjectid] + '/label_predict_ED.nii.gz')
            # # nib.save(predictlabel_ED, savefold)
            # # savefold = os.path.join(datafile[subjectid] + '/label_predict_ES.nii.gz')
            # # nib.save(predictlabel_ES, savefold)

            print(datafile[subjectid])

            subjectfile = glob.glob(datafile[subjectid] + '/patient*_gt.nii.gz')
            imagename_ED = subjectfile[0].replace('_gt.nii.gz', '.nii.gz')
            imagename_ES = subjectfile[1].replace('_gt.nii.gz', '.nii.gz')                        
            predictlabel_ED, predictlabel_ES = ProcessTestDataset_upload(imagename_ED, imagename_ES, G_net)
            savefold = subjectfile[0].replace('_gt.nii.gz', '_predict_249.nii.gz')          
            nib.save(predictlabel_ED, savefold)
            savefold = subjectfile[1].replace('_gt.nii.gz', '_predict_249.nii.gz')   
            nib.save(predictlabel_ES, savefold)

        print(str_for_action + ' end ')

if __name__ == '__main__':
    main()
