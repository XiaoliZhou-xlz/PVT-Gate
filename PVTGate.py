import torch
import torch.nn.functional as F
from torch import nn
from pvtv2 import pvt_v2_b2
from multi_scale_module import PPM
from multi_scale_module import DeepPoolLayer

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

################################CIM#####################################################
class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

#####################################PVTGate##########################################
class PVTGate(nn.Module):

    def __init__(self):
        super(PVTGate, self).__init__()
        self.backbone = pvt_v2_b2()
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        ################################Gate############################################
        self.attention_feature4 = nn.Sequential(nn.Conv2d(64+64, 2, kernel_size=3, padding=1))
        self.attention_feature3 = nn.Sequential(nn.Conv2d(128+64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.attention_feature2 = nn.Sequential(nn.Conv2d(320+128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
                                                nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.attention_feature1 = nn.Sequential(nn.Conv2d(512+320, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU(),
                                                 nn.Conv2d(512, 320, kernel_size=3, padding=1), nn.BatchNorm2d(320), nn.PReLU(),
                                                 nn.Conv2d(320, 2, kernel_size=3, padding=1))
        ###############################Transition Layer####################################
        self.dem1 = nn.Sequential(nn.Conv2d(512, 320, kernel_size=3, padding=1), nn.BatchNorm2d(320), nn.PReLU())
        self.dem2 = nn.Sequential(nn.Conv2d(320, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.dem3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.dem4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        ################################PPM###############################################
        self.ppm = nn.Sequential(PPM(down_dim=512))
        ################################FAD###############################################
        self.fam1 = nn.Sequential(DeepPoolLayer(k=128, k_out=128, need_x2=False, need_fuse=False))
        self.fam2 = nn.Sequential(DeepPoolLayer(k=64, k_out=64, need_x2=False, need_fuse=False))
        self.fam3 = nn.Sequential(DeepPoolLayer(k=64, k_out=64, need_x2=False, need_fuse=False))
        ################################PVT branch#########################################
        self.output1 = nn.Sequential(nn.Conv2d(320, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.output2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.output3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.output4 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))
        #################################CIM###############################################
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        ##################################################################################
        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True


    def forward(self, x):
        input = x
        B,_,_,_ = input.size()
        #####################################PVT Encoder#######################################
        pvt = self.backbone(x)
        E1 = pvt[0]
        E2 = pvt[1]
        E3 = pvt[2]
        E4 = pvt[3]
        ######################################CIM############################################
        E1 = self.ca(E1) * E1  # channel attention
        cim_feature = self.sa(E1) * E1  # spatial attention
        ################################Transition Layer#####################################
        T4 = self.dem1(E4)
        T3 = self.dem2(E3)
        T2 = self.dem3(E2)
        T1 = self.dem4(E1)
        #################################PPM###############################################
        P = self.ppm(E4)
        ################################PVT branch###########################################
        G4 = self.attention_feature1(torch.cat((E4, T4), 1))
        G4 = F.adaptive_avg_pool2d(F.sigmoid(G4), 1)
        D4 = self.output1(G4[:, 0, :, :].unsqueeze(1).repeat(1, 320, 1, 1) * T4)

        G3 = self.attention_feature2(torch.cat((E3,F.upsample(D4, size=E3.size()[2:], mode='bilinear')),1))
        G3 = F.adaptive_avg_pool2d(F.sigmoid(G3),1)
        F1 = self.fam1(F.upsample(D4, size=E3.size()[2:], mode='bilinear') + G3[:, 0, :, :].unsqueeze(1).repeat(1, 128, 1, 1) * T3 + F.upsample(P[:, 0, :, :].unsqueeze(1).repeat(1, 128, 1, 1), size=E3.size()[2:], mode='bilinear')+F.upsample(cim_feature[:, 0, :, :].unsqueeze(1).repeat(1, 128, 1, 1), size=E3.size()[2:], mode='bilinear'))
        D3 = self.output2(F1)

        G2 = self.attention_feature3(torch.cat((E2,F.upsample(D3, size=E2.size()[2:], mode='bilinear')),1))
        G2 = F.adaptive_avg_pool2d(F.sigmoid(G2),1)
        F2 = self.fam2(F.upsample(D3, size=E2.size()[2:], mode='bilinear') + G2[:, 0, :, :].unsqueeze(1).repeat(1, 64, 1, 1) * T2 + F.upsample(P[:, 0, :, :].unsqueeze(1).repeat(1, 64, 1, 1), size=E2.size()[2:], mode='bilinear')+F.upsample(cim_feature, size=E2.size()[2:], mode='bilinear'))
        D2 = self.output3(F2)

        G1 = self.attention_feature4(torch.cat((E1,F.upsample(D2, size=E1.size()[2:], mode='bilinear')),1))
        G1 = F.adaptive_avg_pool2d(F.sigmoid(G1),1)
        F3 = self.fam3(F.upsample(D2, size=E1.size()[2:], mode='bilinear') + G1[:, 0, :, :].unsqueeze(1).repeat(1, 64, 1, 1) * T1 + F.upsample(P[:, 0, :, :].unsqueeze(1).repeat(1, 64, 1, 1), size=E1.size()[2:], mode='bilinear')+F.upsample(cim_feature, size=E1.size()[2:], mode='bilinear'))
        D1 = self.output4(F3)
        ################################output###########################################
        output_pvt = F.upsample(D1, size=input.size()[2:], mode='bilinear')
        output_pvt = F.upsample(output_pvt , size=input.size()[2:], mode='bilinear')
        #######################################################################
        if self.training:
            return output_pvt
        return F.sigmoid(output_pvt)


if __name__ == "__main__":
    model = PVTGate().cuda()
    input = torch.autograd.Variable(torch.randn(4, 3, 384, 384)).cuda()
    output = model(input)

