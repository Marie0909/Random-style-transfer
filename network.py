import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Generator, self).__init__()
        self.E_feature = U_Encoder(in_ch)

        self.D_feature_lab1 = U_Decoder_lab()
        self.D_feature_img1 = U_Decoder_lab()
        self.conv10_Tseg = nn.Conv2d(32, out_ch, 1)
        self.conv10_Timg = nn.Conv2d(32, 1, 1)
        self.cross_task_T = CrossTaskmodule()
        
        self.D_feature_lab2 = U_Decoder_lab()
        self.D_feature_img2 = U_Decoder_lab()
        self.conv10_Sseg = nn.Conv2d(32, 1, 1)
        self.conv10_Simg = nn.Conv2d(32, 1, 1)
        self.cross_task_S = CrossTaskmodule()       

    def forward(self, x, y, MIdex_t, MIdex_s):
        input_s = torch.cat([x, MIdex_s-MIdex_t], dim=1)
        F_target = self.E_feature(input_s)
        c9_Tseg = self.D_feature_lab1(F_target)
        c9_Timg = self.D_feature_img1(F_target)
        c9_Tseg_new = self.cross_task_T(c9_Tseg, c9_Timg)
        c10_Tseg = self.conv10_Tseg(c9_Tseg_new)
        out_Tseg = nn.Softmax(dim=1)(c10_Tseg)
        out_Timg = self.conv10_Timg(c9_Timg)

        input_t = torch.cat([out_Timg, MIdex_t-MIdex_s], dim=1)
        F_source = self.E_feature(input_t)
        c9_Sseg = self.D_feature_lab2(F_source)
        c9_Simg = self.D_feature_img2(F_source)
        c9_Sseg_new = self.cross_task_S(c9_Sseg, c9_Simg)
        c10_Sseg = self.conv10_Sseg(c9_Sseg_new)
        out_Sseg = nn.Softmax(dim=1)(c10_Sseg)
        out_Simg = self.conv10_Simg(c9_Simg)

        return out_Tseg, out_Timg, out_Sseg, out_Simg

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.D_src = nn.Sequential(
            *discriminator_block(1, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

        self.D_cls = nn.Sequential(
            *discriminator_block(1, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 4, 4, padding=1)
        )

    def forward(self, img):
        out_src = self.D_src(img)
        out_src = F.avg_pool2d(out_src, kernel_size=out_src.size()[2:4])
        out_src = nn.Sigmoid()(out_src)

        out_cls = self.D_cls(img)
        out_cls = F.avg_pool2d(out_cls, kernel_size=out_cls.size()[2:4])
        out_cls = nn.Softmax(dim=1)(out_cls)

        return out_src, out_cls

class SRmodule(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(SRmodule, self).__init__()

        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)

        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.conv10 = nn.Conv2d(32,out_ch, 1)


    def forward(self, x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)

        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        out = nn.Softmax(dim=1)(c10)

        return out

class SCmodule(nn.Module):

    def __init__(self):
        super(SCmodule, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(512 ,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,1)
        )


    def forward(self, feature):
        c1, c2, c3, c4, c5=feature
        out = F.avg_pool2d(c5, kernel_size=c5.size()[2:4])
        out= out.view(out.shape[0], -1)
        out_sliceID= self.linear(out)

        return out_sliceID

class CrossTaskmodule(nn.Module):

    def __init__(self, reduction=8):
        super(CrossTaskmodule, self).__init__()

        self.linear1 = nn.Linear(32, 32 // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(32 // reduction, 32, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, F_seg, F_tsl):
        y = F.avg_pool2d(F_tsl, kernel_size=F_tsl.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        F_tsl_new = F_tsl*y
        F_seg_new = F_seg+F_tsl_new

        return F_seg_new

class U_Encoder(nn.Module):
    def __init__(self,in_ch):
        super(U_Encoder, self).__init__()

        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)


    def forward(self, x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        c = c1, c2, c3, c4, c5
        #p = p1, p2, p3, p4

        return c

class U_Decoder_lab(nn.Module):
    def __init__(self):
        super(U_Decoder_lab, self).__init__()

        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)

    def forward(self, feature):
        c1, c2, c3, c4, c5=feature
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)

        return c9

class U_Decoder_img(nn.Module):
    def __init__(self, out_ch):
        super(U_Decoder_img, self).__init__()
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.conv10 = nn.Conv2d(32, out_ch, 1)

    def forward(self, input):
        c1, c2, c3, c4, c5 = input
        up_6 = self.up6(c5)
        c6=self.conv6(up_6)
        up_7=self.up7(c6)
        c7=self.conv7(up_7)
        up_8=self.up8(c7)
        c8=self.conv8(up_8)
        up_9=self.up9(c8)
        c9=self.conv9(up_9)
        c10=self.conv10(c9)

        return c10

class Seg_2DNet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Seg_2DNet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64,out_ch, 1)

        self.SA = SAmodule(1024)
        self.CA = CAmodule(1024)

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)

        # c5_SA = self.SA(c5)
        # c5_CA = self.CA(c5)

        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        out = nn.Softmax(dim=1)(c10)

        return out

class DoubleConv_Leaky(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv_Leaky, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class CAmodule(nn.Module):

    def __init__(self, n_features, reduction=16):
        super(CAmodule, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):

        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        return x * y

class SAmodule(nn.Module):

    def __init__(self, n_features, reduction=8):
        super(SAmodule, self).__init__()

        self.conv1 = nn.Conv2d(n_features, n_features // reduction, 1)
        self.conv2 = nn.Conv2d(n_features, n_features // reduction, 1)
        self.conv3 = nn.Conv2d(n_features, n_features // reduction, 1)
        self.conv4 = nn.Conv2d(n_features // reduction, n_features, 1)
        self.nonlin = nn.Softmax(dim=1)

    def forward(self, x):

        y = x
        b, c, m, n= self.conv1(y).size()
        f = self.conv1(y).reshape(b, c, -1).permute(0, 2, 1)
        g = self.conv2(y)
        beta = self.nonlin(torch.bmm(f, g.reshape(b, c, -1)))

        h = self.conv3(y)
        o = self.conv4(torch.bmm(h.reshape(b, c, -1), beta).reshape(b, c, m, n))

        return o+x

class SegNet_2task(nn.Module):
    def __init__(self, in_ch, out_ch1, out_ch2):
        super(SegNet_2task, self).__init__()       
        self.D_feature_lab1 = U_Decoder_lab()
        # for p in self.parameters():
        #     p.requires_grad=False
        self.D_feature_img1 = U_Decoder_lab()
        self.E_feature = U_Encoder(in_ch)
        
        self.conv10_Tseg = nn.Conv2d(32, out_ch1, 1)
        self.conv10_Timg = nn.Conv2d(32, out_ch1, 1)

        self.cross_task_T = CrossTaskmodule()

        # self.pool = nn.AdaptiveAvgPool2d((2, 2))
        # self.L1 = nn.Linear(512, 256)
        # self.L2 = nn.Linear(256, 128)
        # self.L3 = nn.Linear(128, 64)
        # self.L4 = nn.Linear(64, out_ch2) #vendor, center, ED, ES
        # self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        F_target = self.E_feature(x)
        c9_Tseg = self.D_feature_lab1(F_target)
        c9_Timg = self.D_feature_img1(F_target)
        c9_Tseg_new = self.cross_task_T(c9_Tseg, c9_Timg)
        c10_Tseg = self.conv10_Tseg(c9_Tseg_new)
        out_Tseg = nn.Softmax(dim=1)(c10_Tseg)
        out_Timg = self.conv10_Timg(c9_Timg)

        # c1, c2, c3, c4, c5 = F_target
        # out = self.pool(c5)
        # out= self.L1(out)
        # out= self.relu(out)
        # out = self.L2(out)
        # out = self.relu(out)
        # out = self.L3(out)
        # out = self.relu(out)
        # out_index = self.L4(out)

        return out_Tseg, out_Timg #, out_index


