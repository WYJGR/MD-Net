import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import get_A
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



class PALayer(nn.Module):
    '''........pixel weight uint(PW).......'''

    def __init__(self, channel):
        super( PALayer, self).__init__()

        self.PA = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(64),
            nn.Conv2d(channel // 8, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.PA(x)
        return x * y


class CALayer(nn.Module):
    '''........Channel weight unit(CW).......'''

    def __init__(self, channel):
        super(CALayer,self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.CA = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.CA(y)
        return x * y


class Block(nn.Module):
    '''........dual-layer weight estimation unit.......'''

    def __init__(self, dim, kernel_size):
        super(Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, padding=(kernel_size // 2), bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, padding=(kernel_size // 2), bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.conv1(x)
        DrawGray(res, 'c1.jpg')
        res1 = self.calayer(res)
        DrawGray(res1, 'channel.jpg')
        res2 = self.palayer(res)
        DrawGray(res2, 'pixel.jpg')
        res = res2 + res1
        DrawGray(res, 'p+c.jpg')

        res = self.conv2(res)
        DrawGray(res, 'c2.jpg')
        res = res + x
        DrawGray(res, 'dweu.jpg')
        return res


class GS(nn.Module):
    '''........Group structure for DWEU.......'''

    def __init__(self, dim, kernel_size, blocks):
        super(GS, self).__init__()
        modules = [Block(dim, kernel_size) for _ in range(blocks)]
        self.gs = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gs(x)
        return res


class Branch(nn.Module):
    '''......Branch......'''

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(Branch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.relu(x2)
        x4 = self.IN(x3)
        x5 = self.conv2(x4)

        return x1, x5

class LANet(nn.Module):
    '''......the structure of LANet......'''

    def __init__(self):
        super(LANet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)###

        self.g1 = GS(64, 3, 4)
        self.g2 = GS(64, 3, 3)
        self.g3 = GS(64, 3, 2)
        self.g4 = GS(64, 3, 1)

        self.brabch_3 = Branch(3, 64, 3)
        self.brabch_5 = Branch(3, 64, 5)
        self.brabch_7 = Branch(3, 64, 7)

        self.fusion = nn.Sequential(*[
            nn.Conv2d(64 * 3, 64 // 8, 3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64 // 8, 3, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])

        self.Final = nn.Conv2d(64, 3, 1, padding=0, bias=True)#####


    def forward(self, x):
        '''.....three branch.......'''
        x11, x1 = self.brabch_3(x)
        x22, x2 = self.brabch_5(x)
        x33, x3 = self.brabch_7(x)
        DrawGray(x11, 'i11.jpg')
        DrawGray(x22, 'i22.jpg')
        DrawGray(x33, 'i33.jpg')
        DrawGray(x1, 'i1.jpg')
        DrawGray(x2, 'i2.jpg')
        DrawGray(x3, 'i3.jpg')

        '''......Multi-layer Fusion......'''
        concate=torch.cat([x1, x2, x3], dim=1)
        DrawGray(concate,'concate.jpg')
        w = self.fusion(torch.cat([x1, x2, x3], dim=1))
        DrawGray(w,'conv2.jpg')
        w = torch.split(w, 1, dim=1)
        x4 = w[0] * x1 + w[1] * x2 + w[2] * x3
        m1=w[0] * x1
        m2 = w[1] * x2
        m3 = w[2] * x3
        DrawGray(w[0],'w1.jpg')
        DrawGray(w[1], 'w2.jpg')
        DrawGray(w[2], 'w3.jpg')
        DrawGray(x4, 'mfm.jpg')
        DrawGray(m1, '1.jpg')
        DrawGray(m2, '2.jpg')
        DrawGray(m3, '3.jpg')
        res1 = self.g1(x4)
        res2 = self.g2(x4)
        res3 = self.g3(x4)
        res4 = self.g4(x4)
        DrawGray(res1, 'dw4.jpg')
        DrawGray(res2, 'dw3.jpg')
        DrawGray(res3, 'dw2.jpg')
        DrawGray(res4, 'dw1.jpg')

        x5 = self.avg_pool(x4)

        res2 = x5 * res1 + x33
        y_hat = self.Final(res2)

        return y_hat

class net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_net = LANet()
        self.mask_net = LANet()

    def forward(self, data):
        j_out = self.image_net(data)
        t_out = self.mask_net(data)


        a_out = get_A(data).cuda()
        I_rec = j_out * t_out + (1 - t_out) * a_out

        return j_out, t_out, I_rec

def DrawGray(outputs, f):
    # 将输出的6个通道合并成一个
    merged_output = torch.sum(outputs, dim=1)

    # 将PyTorch张量转换为NumPy数组
    merged_output_np = merged_output.cpu().numpy()
    #merged_output_np = merged_output.detach().cpu().numpy()

    # 绘制合并后的输出
    plt.imshow(merged_output_np[0], cmap='gray', aspect='auto', extent=[0, 256, 256, 0])  # 假设是灰度图像
    plt.axis('off')  # 关闭坐标轴
    # plt.show()
    plt.savefig(f)


