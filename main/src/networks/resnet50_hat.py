import sys
import torch
import numpy as np

import utils

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla):
        super(Net,self).__init__()
        ncha,size,_=inputsize
        self.taskcla=taskcla
        #self.taskcla=[(0,2),(0,2),(0,2)]
        #self.maxpool = torch.nn.MaxPool2d(kernel_size=7, stride=2, padding=1)
        #self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.relu=torch.nn.ReLU()

        #self.c1=torch.nn.Conv2d(ncha, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.c1=torch.nn.Conv2d(ncha, 64, kernel_size=3, stride=1, padding=1,bias=False)
        #layer2-1
        self.c2_1_1=torch.nn.Conv2d(64,64,kernel_size=1, stride=1, bias=False)
        self.c2_1_2=torch.nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1,bias=False)
        self.c2_1_3=torch.nn.Conv2d(64,256,kernel_size=1, stride=1, bias=False)
        self.c2_1_d=torch.nn.Conv2d(64,256,kernel_size=1, stride=1, bias=False)
        #2-2
        self.c2_2_1=torch.nn.Conv2d(256,64,kernel_size=1, stride=1, bias=False)
        self.c2_2_2=torch.nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1,bias=False)
        self.c2_2_3=torch.nn.Conv2d(64,256,kernel_size=1, stride=1, bias=False)
        #2-3
        self.c2_3_1=torch.nn.Conv2d(256,64,kernel_size=1, stride=1, bias=False)
        self.c2_3_2=torch.nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1,bias=False)
        self.c2_3_3=torch.nn.Conv2d(64,256,kernel_size=1, stride=1, bias=False)
        #layer3-1
        self.c3_1_1=torch.nn.Conv2d(256,128,kernel_size=1, stride=1, bias=False)
        self.c3_1_2=torch.nn.Conv2d(128,128,kernel_size=3, stride=2, padding=1,bias=False)
        self.c3_1_3=torch.nn.Conv2d(128,512,kernel_size=1, stride=1, bias=False)
        self.c3_1_d=torch.nn.Conv2d(256,512,kernel_size=1, stride=2, bias=False)
        #3-2
        self.c3_2_1=torch.nn.Conv2d(512,128,kernel_size=1, stride=1, bias=False)
        self.c3_2_2=torch.nn.Conv2d(128,128,kernel_size=3, stride=1, padding=1,bias=False)
        self.c3_2_3=torch.nn.Conv2d(128,512,kernel_size=1, stride=1, bias=False)
        #3-3
        self.c3_3_1=torch.nn.Conv2d(512,128,kernel_size=1, stride=1, bias=False)
        self.c3_3_2=torch.nn.Conv2d(128,128,kernel_size=3, stride=1, padding=1,bias=False)
        self.c3_3_3=torch.nn.Conv2d(128,512,kernel_size=1, stride=1, bias=False)
        #3-4
        self.c3_4_1=torch.nn.Conv2d(512,128,kernel_size=1, stride=1, bias=False)
        self.c3_4_2=torch.nn.Conv2d(128,128,kernel_size=3, stride=1, padding=1,bias=False)
        self.c3_4_3=torch.nn.Conv2d(128,512,kernel_size=1, stride=1, bias=False)
        #layer4-1
        self.c4_1_1=torch.nn.Conv2d(512,256,kernel_size=1, stride=1, bias=False)
        self.c4_1_2=torch.nn.Conv2d(256,256,kernel_size=3, stride=2, padding=1,bias=False)
        self.c4_1_3=torch.nn.Conv2d(256,1024,kernel_size=1, stride=1, bias=False)
        self.c4_1_d=torch.nn.Conv2d(512,1024,kernel_size=1, stride=2, bias=False)
        #4-2
        self.c4_2_1=torch.nn.Conv2d(1024,256,kernel_size=1, stride=1, bias=False)
        self.c4_2_2=torch.nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1,bias=False)
        self.c4_2_3=torch.nn.Conv2d(256,1024,kernel_size=1, stride=1, bias=False)
        #4-3
        self.c4_3_1=torch.nn.Conv2d(1024,256,kernel_size=1, stride=1, bias=False)
        self.c4_3_2=torch.nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1,bias=False)
        self.c4_3_3=torch.nn.Conv2d(256,1024,kernel_size=1, stride=1, bias=False)
        #4-4
        self.c4_4_1=torch.nn.Conv2d(1024,256,kernel_size=1, stride=1, bias=False)
        self.c4_4_2=torch.nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1,bias=False)
        self.c4_4_3=torch.nn.Conv2d(256,1024,kernel_size=1, stride=1, bias=False)
        #4-5
        self.c4_5_1=torch.nn.Conv2d(1024,256,kernel_size=1, stride=1, bias=False)
        self.c4_5_2=torch.nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1,bias=False)
        self.c4_5_3=torch.nn.Conv2d(256,1024,kernel_size=1, stride=1, bias=False)
        #4-6
        self.c4_6_1=torch.nn.Conv2d(1024,256,kernel_size=1, stride=1, bias=False)
        self.c4_6_2=torch.nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1,bias=False)
        self.c4_6_3=torch.nn.Conv2d(256,1024,kernel_size=1, stride=1, bias=False)

        #layer5-1
        self.c5_1_1=torch.nn.Conv2d(1024,512,kernel_size=1, stride=1, bias=False)
        self.c5_1_2=torch.nn.Conv2d(512,512,kernel_size=3, stride=2, padding=1,bias=False)
        self.c5_1_3=torch.nn.Conv2d(512,2048,kernel_size=1, stride=1, bias=False)
        self.c5_1_d=torch.nn.Conv2d(1024,2048,kernel_size=1, stride=2, bias=False)
        #5-2
        self.c5_2_1=torch.nn.Conv2d(2048,512,kernel_size=1, stride=1, bias=False)
        self.c5_2_2=torch.nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1,bias=False)
        self.c5_2_3=torch.nn.Conv2d(512,2048,kernel_size=1, stride=1, bias=False)
        #5-3
        self.c5_3_1=torch.nn.Conv2d(2048,512,kernel_size=1, stride=1, bias=False)
        self.c5_3_2=torch.nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1,bias=False)
        self.c5_3_3=torch.nn.Conv2d(512,2048,kernel_size=1, stride=1, bias=False)
        
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(2048,n))#need to be modified
        #self.last=torch.nn.Linear(2048,100)#100
        
        self.gate=torch.nn.Sigmoid()
        em_word=len(self.taskcla)#1
        # All embedding stuff should start with 'e'
        self.ec1=torch.nn.Embedding(em_word,64)
        #2-1
        self.ec2_1_1=torch.nn.Embedding(em_word,64)
        self.ec2_1_2=torch.nn.Embedding(em_word,64)
        self.ec2_1_3=torch.nn.Embedding(em_word,256)
        self.ec2_1_d=torch.nn.Embedding(em_word,256)
        self.ect2_1=torch.nn.Embedding(em_word,256)
        #2-2
        self.ec2_2_1=torch.nn.Embedding(em_word,64)
        self.ec2_2_2=torch.nn.Embedding(em_word,64)
        self.ec2_2_3=torch.nn.Embedding(em_word,256)
        #2-3
        self.ec2_3_1=torch.nn.Embedding(em_word,64)
        self.ec2_3_2=torch.nn.Embedding(em_word,64)
        self.ec2_3_3=torch.nn.Embedding(em_word,256)
        #3-1
        self.ec3_1_1=torch.nn.Embedding(em_word,128)
        self.ec3_1_2=torch.nn.Embedding(em_word,128)
        self.ec3_1_3=torch.nn.Embedding(em_word,512)
        self.ec3_1_d=torch.nn.Embedding(em_word,512)
        self.ect3_1=torch.nn.Embedding(em_word,512)
        #3-2
        self.ec3_2_1=torch.nn.Embedding(em_word,128)
        self.ec3_2_2=torch.nn.Embedding(em_word,128)
        self.ec3_2_3=torch.nn.Embedding(em_word,512)
        #3-3
        self.ec3_3_1=torch.nn.Embedding(em_word,128)
        self.ec3_3_2=torch.nn.Embedding(em_word,128)
        self.ec3_3_3=torch.nn.Embedding(em_word,512)
        #3-4
        self.ec3_4_1=torch.nn.Embedding(em_word,128)
        self.ec3_4_2=torch.nn.Embedding(em_word,128)
        self.ec3_4_3=torch.nn.Embedding(em_word,512)
        #4-1
        self.ec4_1_1=torch.nn.Embedding(em_word,256)
        self.ec4_1_2=torch.nn.Embedding(em_word,256)
        self.ec4_1_3=torch.nn.Embedding(em_word,1024)
        self.ec4_1_d=torch.nn.Embedding(em_word,1024)
        self.ect4_1=torch.nn.Embedding(em_word,1024)
        #4-2
        self.ec4_2_1=torch.nn.Embedding(em_word,256)
        self.ec4_2_2=torch.nn.Embedding(em_word,256)
        self.ec4_2_3=torch.nn.Embedding(em_word,1024)
        #4-3
        self.ec4_3_1=torch.nn.Embedding(em_word,256)
        self.ec4_3_2=torch.nn.Embedding(em_word,256)
        self.ec4_3_3=torch.nn.Embedding(em_word,1024)
        #4-4
        self.ec4_4_1=torch.nn.Embedding(em_word,256)
        self.ec4_4_2=torch.nn.Embedding(em_word,256)
        self.ec4_4_3=torch.nn.Embedding(em_word,1024)
        #4-5
        self.ec4_5_1=torch.nn.Embedding(em_word,256)
        self.ec4_5_2=torch.nn.Embedding(em_word,256)
        self.ec4_5_3=torch.nn.Embedding(em_word,1024)
        #4-6
        self.ec4_6_1=torch.nn.Embedding(em_word,256)
        self.ec4_6_2=torch.nn.Embedding(em_word,256)
        self.ec4_6_3=torch.nn.Embedding(em_word,1024)
        #5-1
        self.ec5_1_1=torch.nn.Embedding(em_word,512)
        self.ec5_1_2=torch.nn.Embedding(em_word,512)
        self.ec5_1_3=torch.nn.Embedding(em_word,2048)
        self.ec5_1_d=torch.nn.Embedding(em_word,2048)
        self.ect5_1=torch.nn.Embedding(em_word,2048)
        #5-2
        self.ec5_2_1=torch.nn.Embedding(em_word,512)
        self.ec5_2_2=torch.nn.Embedding(em_word,512)
        self.ec5_2_3=torch.nn.Embedding(em_word,2048)
        #5-3
        self.ec5_3_1=torch.nn.Embedding(em_word,512)
        self.ec5_3_2=torch.nn.Embedding(em_word,512)
        self.ec5_3_3=torch.nn.Embedding(em_word,2048)

        #bn 1-1
        #self.bn=torch.nn.BatchNorm2d(64,affine=False)
        self.bn1=torch.nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        #2-1
        self.bn2_1_1=torch.nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.bn2_1_2=torch.nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.bn2_1_3=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.bn2_1_d=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        #self.bnt2_1=torch.nn.Embedding(len(self.taskcla),256)
        #2-2
        self.bn2_2_1=torch.nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.bn2_2_2=torch.nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.bn2_2_3=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        #2-3
        self.bn2_3_1=torch.nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.bn2_3_2=torch.nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.bn2_3_3=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        #3-1
        self.bn3_1_1=torch.nn.BatchNorm2d(128, affine=False, track_running_stats=False)
        self.bn3_1_2=torch.nn.BatchNorm2d(128, affine=False, track_running_stats=False)
        self.bn3_1_3=torch.nn.BatchNorm2d(512, affine=False, track_running_stats=False)
        self.bn3_1_d=torch.nn.BatchNorm2d(512, affine=False, track_running_stats=False)
        #self.ect3_1=torch.nn.Embedding(len(self.taskcla),512)
        #3-2
        self.bn3_2_1=torch.nn.BatchNorm2d(128, affine=False, track_running_stats=False)
        self.bn3_2_2=torch.nn.BatchNorm2d(128, affine=False, track_running_stats=False)
        self.bn3_2_3=torch.nn.BatchNorm2d(512, affine=False, track_running_stats=False)
        #3-3
        self.bn3_3_1=torch.nn.BatchNorm2d(128, affine=False, track_running_stats=False)
        self.bn3_3_2=torch.nn.BatchNorm2d(128, affine=False, track_running_stats=False)
        self.bn3_3_3=torch.nn.BatchNorm2d(512, affine=False, track_running_stats=False)
        #3-4
        self.bn3_4_1=torch.nn.BatchNorm2d(128, affine=False, track_running_stats=False)
        self.bn3_4_2=torch.nn.BatchNorm2d(128, affine=False, track_running_stats=False)
        self.bn3_4_3=torch.nn.BatchNorm2d(512, affine=False, track_running_stats=False)
        #4-1
        self.bn4_1_1=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.bn4_1_2=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.bn4_1_3=torch.nn.BatchNorm2d(1024, affine=False, track_running_stats=False)
        self.bn4_1_d=torch.nn.BatchNorm2d(1024, affine=False, track_running_stats=False)
        #self.ect3_1=torch.nn.Embedding(len(self.taskcla),512)
        #4-2
        self.bn4_2_1=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.bn4_2_2=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.bn4_2_3=torch.nn.BatchNorm2d(1024, affine=False, track_running_stats=False)
        #4-3
        self.bn4_3_1=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.bn4_3_2=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.bn4_3_3=torch.nn.BatchNorm2d(1024, affine=False, track_running_stats=False)
        #4-4
        self.bn4_4_1=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.bn4_4_2=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.bn4_4_3=torch.nn.BatchNorm2d(1024, affine=False, track_running_stats=False)
        #4-5
        self.bn4_5_1=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.bn4_5_2=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.bn4_5_3=torch.nn.BatchNorm2d(1024, affine=False, track_running_stats=False)
        #4-6
        self.bn4_6_1=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.bn4_6_2=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.bn4_6_3=torch.nn.BatchNorm2d(1024, affine=False, track_running_stats=False)
        #5-1
        self.bn5_1_1=torch.nn.BatchNorm2d(512, affine=False, track_running_stats=False)
        self.bn5_1_2=torch.nn.BatchNorm2d(512, affine=False, track_running_stats=False)
        self.bn5_1_3=torch.nn.BatchNorm2d(2048, affine=False, track_running_stats=False)
        self.bn5_1_d=torch.nn.BatchNorm2d(2048, affine=False, track_running_stats=False)
        #self.ect3_1=torch.nn.Embedding(len(self.taskcla),512)
        #5-2
        self.bn5_2_1=torch.nn.BatchNorm2d(512, affine=False, track_running_stats=False)
        self.bn5_2_2=torch.nn.BatchNorm2d(512, affine=False, track_running_stats=False)
        self.bn5_2_3=torch.nn.BatchNorm2d(2048, affine=False, track_running_stats=False)
        #5-3
        self.bn5_3_1=torch.nn.BatchNorm2d(512, affine=False, track_running_stats=False)
        self.bn5_3_2=torch.nn.BatchNorm2d(512, affine=False, track_running_stats=False)
        self.bn5_3_3=torch.nn.BatchNorm2d(2048, affine=False, track_running_stats=False)

        return

    def forward(self,t,x,s=1):
    #def forward(self,x):
        # Gates
        masks=self.mask(t,s=s)

        gc1,gc2_1_1,gc2_1_2,gc2_1_3,gc2_1_d,gct2_1,gc2_2_1,gc2_2_2,gc2_2_3,gc2_3_1,gc2_3_2,gc2_3_3,\
        gc3_1_1,gc3_1_2,gc3_1_3,gc3_1_d,gct3_1,gc3_2_1,gc3_2_2,gc3_2_3,gc3_3_1,gc3_3_2,gc3_3_3,gc3_4_1,gc3_4_2,gc3_4_3,\
        gc4_1_1,gc4_1_2,gc4_1_3,gc4_1_d,gct4_1,gc4_2_1,gc4_2_2,gc4_2_3,gc4_3_1,gc4_3_2,gc4_3_3,gc4_4_1,gc4_4_2,gc4_4_3,\
        gc4_5_1,gc4_5_2,gc4_5_3,gc4_6_1,gc4_6_2,gc4_6_3,\
        gc5_1_1,gc5_1_2,gc5_1_3,gc5_1_d,gct5_1,gc5_2_1,gc5_2_2,gc5_2_3,gc5_3_1,gc5_3_2,gc5_3_3=masks
        
        #layer1
        #h=self.maxpool(self.relu(self.bn1(self.c1(x))))
        h=self.relu(self.bn1(self.c1(x)))
        h=h*gc1.view(1,-1,1,1).expand_as(h)
        #layer2-1
        identity=h
        h=self.relu(self.bn2_1_1(self.c2_1_1(h)));h=h*gc2_1_1.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.bn2_1_2(self.c2_1_2(h)));h=h*gc2_1_2.view(1,-1,1,1).expand_as(h)
        h=self.bn2_1_3(self.c2_1_3(h));h=h*gc2_1_3.view(1,-1,1,1).expand_as(h)

        identity=self.bn2_1_d(self.c2_1_d(identity))
        identity=identity*gc2_1_d.view(1,-1,1,1).expand_as(identity)

        h+=identity
        h=self.relu(h);h=h*gct2_1.view(1,-1,1,1).expand_as(h)
        #2-2
        identity=h
        h=self.relu(self.bn2_2_1(self.c2_2_1(h)));h=h*gc2_2_1.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.bn2_2_2(self.c2_2_2(h)));h=h*gc2_2_2.view(1,-1,1,1).expand_as(h)
        h=self.bn2_2_3(self.c2_2_3(h));h=h*gc2_2_3.view(1,-1,1,1).expand_as(h)

        h+=identity;h=self.relu(h)
        #2-3
        identity=h
        h=self.relu(self.bn2_3_1(self.c2_3_1(h)));h=h*gc2_3_1.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.bn2_3_2(self.c2_3_2(h)));h=h*gc2_3_2.view(1,-1,1,1).expand_as(h)
        h=self.bn2_3_3(self.c2_3_3(h));h=h*gc2_3_3.view(1,-1,1,1).expand_as(h)

        h+=identity;h=self.relu(h)
        #3-1
        identity=h
        h=self.relu(self.bn3_1_1(self.c3_1_1(h)));h=h*gc3_1_1.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.bn3_1_2(self.c3_1_2(h)));h=h*gc3_1_2.view(1,-1,1,1).expand_as(h)
        h=self.bn3_1_3(self.c3_1_3(h));h=h*gc3_1_3.view(1,-1,1,1).expand_as(h)

        identity=self.bn3_1_d(self.c3_1_d(identity))
        identity=identity*gc3_1_d.view(1,-1,1,1).expand_as(identity)#;print('+1l+',identity.shape)

        h+=identity
        h=self.relu(h);h=h*gct3_1.view(1,-1,1,1).expand_as(h)
        #3-2
        identity=h
        h=self.relu(self.bn3_2_1(self.c3_2_1(h)));h=h*gc3_2_1.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.bn3_2_2(self.c3_2_2(h)));h=h*gc3_2_2.view(1,-1,1,1).expand_as(h)
        h=self.bn3_2_3(self.c3_2_3(h));h=h*gc3_2_3.view(1,-1,1,1).expand_as(h)

        h+=identity;h=self.relu(h)
        #3-3
        identity=h
        h=self.relu(self.bn3_3_1(self.c3_3_1(h)));h=h*gc3_3_1.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.bn3_3_2(self.c3_3_2(h)));h=h*gc3_3_2.view(1,-1,1,1).expand_as(h)
        h=self.bn3_3_3(self.c3_3_3(h));h=h*gc3_3_3.view(1,-1,1,1).expand_as(h)

        h+=identity;h=self.relu(h)
        #3-4
        identity=h
        h=self.relu(self.bn3_4_1(self.c3_4_1(h)));h=h*gc3_4_1.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.bn3_4_2(self.c3_4_2(h)));h=h*gc3_4_2.view(1,-1,1,1).expand_as(h)
        h=self.bn3_4_3(self.c3_4_3(h));h=h*gc3_4_3.view(1,-1,1,1).expand_as(h)

        h+=identity;h=self.relu(h)
        
        #4-1
        identity=h
        h=self.relu(self.bn4_1_1(self.c4_1_1(h)));h=h*gc4_1_1.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.bn4_1_2(self.c4_1_2(h)));h=h*gc4_1_2.view(1,-1,1,1).expand_as(h)
        h=self.bn4_1_3(self.c4_1_3(h));h=h*gc4_1_3.view(1,-1,1,1).expand_as(h)

        identity=self.bn4_1_d(self.c4_1_d(identity))
        identity=identity*gc4_1_d.view(1,-1,1,1).expand_as(identity)#;print('+1l+',identity.shape)

        h+=identity
        h=self.relu(h);h=h*gct4_1.view(1,-1,1,1).expand_as(h)
        #4-2
        identity=h
        h=self.relu(self.bn4_2_1(self.c4_2_1(h)));h=h*gc4_2_1.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.bn4_2_2(self.c4_2_2(h)));h=h*gc4_2_2.view(1,-1,1,1).expand_as(h)
        h=self.bn4_2_3(self.c4_2_3(h));h=h*gc4_2_3.view(1,-1,1,1).expand_as(h)

        h+=identity;h=self.relu(h)
        #4-3
        identity=h
        h=self.relu(self.bn4_3_1(self.c4_3_1(h)));h=h*gc4_3_1.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.bn4_3_2(self.c4_3_2(h)));h=h*gc4_3_2.view(1,-1,1,1).expand_as(h)
        h=self.bn4_3_3(self.c4_3_3(h));h=h*gc4_3_3.view(1,-1,1,1).expand_as(h)

        h+=identity;h=self.relu(h)
        #4-4
        identity=h
        h=self.relu(self.bn4_4_1(self.c4_4_1(h)));h=h*gc4_4_1.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.bn4_4_2(self.c4_4_2(h)));h=h*gc4_4_2.view(1,-1,1,1).expand_as(h)
        h=self.bn4_4_3(self.c4_4_3(h));h=h*gc4_4_3.view(1,-1,1,1).expand_as(h)

        h+=identity;h=self.relu(h)
        #4-5
        identity=h
        h=self.relu(self.bn4_5_1(self.c4_5_1(h)));h=h*gc4_5_1.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.bn4_5_2(self.c4_5_2(h)));h=h*gc4_5_2.view(1,-1,1,1).expand_as(h)
        h=self.bn4_5_3(self.c4_5_3(h));h=h*gc4_5_3.view(1,-1,1,1).expand_as(h)

        h+=identity;h=self.relu(h)

        #4-6
        identity=h
        h=self.relu(self.bn4_6_1(self.c4_6_1(h)));h=h*gc4_6_1.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.bn4_6_2(self.c4_6_2(h)));h=h*gc4_6_2.view(1,-1,1,1).expand_as(h)
        h=self.bn4_6_3(self.c4_6_3(h));h=h*gc4_6_3.view(1,-1,1,1).expand_as(h)

        h+=identity;h=self.relu(h)

        #5-1
        identity=h
        h=self.relu(self.bn5_1_1(self.c5_1_1(h)));h=h*gc5_1_1.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.bn5_1_2(self.c5_1_2(h)));h=h*gc5_1_2.view(1,-1,1,1).expand_as(h)
        h=self.bn5_1_3(self.c5_1_3(h));h=h*gc5_1_3.view(1,-1,1,1).expand_as(h)

        identity=self.bn5_1_d(self.c5_1_d(identity))
        identity=identity*gc5_1_d.view(1,-1,1,1).expand_as(identity)

        h+=identity
        h=self.relu(h);h=h*gct5_1.view(1,-1,1,1).expand_as(h)
        #5-2
        identity=h
        h=self.relu(self.bn5_2_1(self.c5_2_1(h)));h=h*gc5_2_1.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.bn5_2_2(self.c5_2_2(h)));h=h*gc5_2_2.view(1,-1,1,1).expand_as(h)
        h=self.bn5_2_3(self.c5_2_3(h));h=h*gc5_2_3.view(1,-1,1,1).expand_as(h)

        h+=identity;h=self.relu(h)
        #5-3
        identity=h
        h=self.relu(self.bn5_3_1(self.c5_3_1(h)));h=h*gc5_3_1.view(1,-1,1,1).expand_as(h)
        h=self.relu(self.bn5_3_2(self.c5_3_2(h)));h=h*gc5_3_2.view(1,-1,1,1).expand_as(h)
        h=self.bn5_3_3(self.c5_3_3(h));h=h*gc5_3_3.view(1,-1,1,1).expand_as(h)

        h+=identity;h=self.relu(h)

        #final
        h = self.avgpool(h)
        h = torch.flatten(h, 1)

        y=[]
        for i,_ in self.taskcla:
            y.append(self.last[i](h))
            #y.append(self.last(h))
        #y=self.last(h)
        return y,masks

    def mask(self,t,s=1):
        em_id=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)#t=0
        gc1=self.gate(s*self.ec1(em_id))
        #2-1
        gc2_1_1=self.gate(s*self.ec2_1_1(em_id))
        gc2_1_2=self.gate(s*self.ec2_1_2(em_id))
        gc2_1_3=self.gate(s*self.ec2_1_3(em_id))
        gc2_1_d=self.gate(s*self.ec2_1_d(em_id))
        gct2_1=self.gate(s*self.ect2_1(em_id))
        #2-2
        gc2_2_1=self.gate(s*self.ec2_2_1(em_id))
        gc2_2_2=self.gate(s*self.ec2_2_2(em_id))
        gc2_2_3=self.gate(s*self.ec2_2_3(em_id))
        #2-3
        gc2_3_1=self.gate(s*self.ec2_3_1(em_id))
        gc2_3_2=self.gate(s*self.ec2_3_2(em_id))
        gc2_3_3=self.gate(s*self.ec2_3_3(em_id))
        #3-1
        gc3_1_1=self.gate(s*self.ec3_1_1(em_id))
        gc3_1_2=self.gate(s*self.ec3_1_2(em_id))
        gc3_1_3=self.gate(s*self.ec3_1_3(em_id))
        gc3_1_d=self.gate(s*self.ec3_1_d(em_id))
        gct3_1=self.gate(s*self.ect3_1(em_id))
        #3-2
        gc3_2_1=self.gate(s*self.ec3_2_1(em_id))
        gc3_2_2=self.gate(s*self.ec3_2_2(em_id))
        gc3_2_3=self.gate(s*self.ec3_2_3(em_id))
        #3-3
        gc3_3_1=self.gate(s*self.ec3_3_1(em_id))
        gc3_3_2=self.gate(s*self.ec3_3_2(em_id))
        gc3_3_3=self.gate(s*self.ec3_3_3(em_id))
        #3-4
        gc3_4_1=self.gate(s*self.ec3_4_1(em_id))
        gc3_4_2=self.gate(s*self.ec3_4_2(em_id))
        gc3_4_3=self.gate(s*self.ec3_4_3(em_id))
        #4-1
        gc4_1_1=self.gate(s*self.ec4_1_1(em_id))
        gc4_1_2=self.gate(s*self.ec4_1_2(em_id))
        gc4_1_3=self.gate(s*self.ec4_1_3(em_id))
        gc4_1_d=self.gate(s*self.ec4_1_d(em_id))
        gct4_1=self.gate(s*self.ect4_1(em_id))
        #4-2
        gc4_2_1=self.gate(s*self.ec4_2_1(em_id))
        gc4_2_2=self.gate(s*self.ec4_2_2(em_id))
        gc4_2_3=self.gate(s*self.ec4_2_3(em_id))
        #4-3
        gc4_3_1=self.gate(s*self.ec4_3_1(em_id))
        gc4_3_2=self.gate(s*self.ec4_3_2(em_id))
        gc4_3_3=self.gate(s*self.ec4_3_3(em_id))
        #4-4
        gc4_4_1=self.gate(s*self.ec4_4_1(em_id))
        gc4_4_2=self.gate(s*self.ec4_4_2(em_id))
        gc4_4_3=self.gate(s*self.ec4_4_3(em_id))
        #4-5
        gc4_5_1=self.gate(s*self.ec4_5_1(em_id))
        gc4_5_2=self.gate(s*self.ec4_5_2(em_id))
        gc4_5_3=self.gate(s*self.ec4_5_3(em_id))
        #4-6
        gc4_6_1=self.gate(s*self.ec4_6_1(em_id))
        gc4_6_2=self.gate(s*self.ec4_6_2(em_id))
        gc4_6_3=self.gate(s*self.ec4_6_3(em_id))

        #5-1
        gc5_1_1=self.gate(s*self.ec5_1_1(em_id))
        gc5_1_2=self.gate(s*self.ec5_1_2(em_id))
        gc5_1_3=self.gate(s*self.ec5_1_3(em_id))
        gc5_1_d=self.gate(s*self.ec5_1_d(em_id))
        gct5_1=self.gate(s*self.ect5_1(em_id))
        #5-2
        gc5_2_1=self.gate(s*self.ec5_2_1(em_id))
        gc5_2_2=self.gate(s*self.ec5_2_2(em_id))
        gc5_2_3=self.gate(s*self.ec5_2_3(em_id))
        #5-3
        gc5_3_1=self.gate(s*self.ec5_3_1(em_id))
        gc5_3_2=self.gate(s*self.ec5_3_2(em_id))
        gc5_3_3=self.gate(s*self.ec5_3_3(em_id))

        return[gc1,gc2_1_1,gc2_1_2,gc2_1_3,gc2_1_d,gct2_1,gc2_2_1,gc2_2_2,gc2_2_3,gc2_3_1,gc2_3_2,gc2_3_3,\
        gc3_1_1,gc3_1_2,gc3_1_3,gc3_1_d,gct3_1,gc3_2_1,gc3_2_2,gc3_2_3,gc3_3_1,gc3_3_2,gc3_3_3,gc3_4_1,gc3_4_2,gc3_4_3,\
        gc4_1_1,gc4_1_2,gc4_1_3,gc4_1_d,gct4_1,gc4_2_1,gc4_2_2,gc4_2_3,gc4_3_1,gc4_3_2,gc4_3_3,gc4_4_1,gc4_4_2,gc4_4_3,\
        gc4_5_1,gc4_5_2,gc4_5_3,gc4_6_1,gc4_6_2,gc4_6_3,\
        gc5_1_1,gc5_1_2,gc5_1_3,gc5_1_d,gct5_1,gc5_2_1,gc5_2_2,gc5_2_3,gc5_3_1,gc5_3_2,gc5_3_3]

    def get_view_for(self,n,masks):        
        gc1,gc2_1_1,gc2_1_2,gc2_1_3,gc2_1_d,gct2_1,gc2_2_1,gc2_2_2,gc2_2_3,gc2_3_1,gc2_3_2,gc2_3_3,\
        gc3_1_1,gc3_1_2,gc3_1_3,gc3_1_d,gct3_1,gc3_2_1,gc3_2_2,gc3_2_3,gc3_3_1,gc3_3_2,gc3_3_3,gc3_4_1,gc3_4_2,gc3_4_3,\
        gc4_1_1,gc4_1_2,gc4_1_3,gc4_1_d,gct4_1,gc4_2_1,gc4_2_2,gc4_2_3,gc4_3_1,gc4_3_2,gc4_3_3,gc4_4_1,gc4_4_2,gc4_4_3,\
        gc4_5_1,gc4_5_2,gc4_5_3,gc4_6_1,gc4_6_2,gc4_6_3,\
        gc5_1_1,gc5_1_2,gc5_1_3,gc5_1_d,gct5_1,gc5_2_1,gc5_2_2,gc5_2_3,gc5_3_1,gc5_3_2,gc5_3_3=masks

        if n=='c1.weight':
            return gc1.data.view(-1,1,1,1).expand_as(self.c1.weight)
        elif n=='c1.bias':
            return gc1.data.view(-1)
        #2-1
        elif n=='c2_1_1.weight':
            post=gc2_1_1.data.view(-1,1,1,1).expand_as(self.c2_1_1.weight)
            pre=gc1.data.view(1,-1,1,1).expand_as(self.c2_1_1.weight)
            return torch.min(post,pre)
        elif n=='c2_1_1.bias':
            return gc2_1_1.data.view(-1)
        elif n=='c2_1_2.weight':
            post=gc2_1_2.data.view(-1,1,1,1).expand_as(self.c2_1_2.weight)
            pre=gc2_1_1.data.view(1,-1,1,1).expand_as(self.c2_1_2.weight)
            return torch.min(post,pre)
        elif n=='c2_1_2.bias':
            return gc2_1_2.data.view(-1)
        elif n=='c2_1_3.weight':
            post=gc2_1_3.data.view(-1,1,1,1).expand_as(self.c2_1_3.weight)
            pre=gc2_1_2.data.view(1,-1,1,1).expand_as(self.c2_1_3.weight)
            return torch.min(post,pre)
        elif n=='c2_1_3.bias':
            return gc2_1_3.data.view(-1)
        elif n=='c2_1_d.weight':
            post=gc2_1_d.data.view(-1,1,1,1).expand_as(self.c2_1_d.weight)
            pre=gc1.data.view(1,-1,1,1).expand_as(self.c2_1_d.weight)
            return torch.min(post,pre)
        elif n=='c2_1_d.bias':
            return gc2_1_d.data.view(-1)
        #2-2
        elif n=='c2_2_1.weight':
            post=gc2_2_1.data.view(-1,1,1,1).expand_as(self.c2_2_1.weight)
            pre=gct2_1.data.view(1,-1,1,1).expand_as(self.c2_2_1.weight)
            return torch.min(post,pre)
        elif n=='c2_2_1.bias':
            return gc2_2_1.data.view(-1)
        elif n=='c2_2_2.weight':
            post=gc2_2_2.data.view(-1,1,1,1).expand_as(self.c2_2_2.weight)
            pre=gc2_2_1.data.view(1,-1,1,1).expand_as(self.c2_2_2.weight)
            return torch.min(post,pre)
        elif n=='c2_2_2.bias':
            return gc2_2_2.data.view(-1)
        elif n=='c2_2_3.weight':
            post=gc2_2_3.data.view(-1,1,1,1).expand_as(self.c2_2_3.weight)
            pre=gc2_2_2.data.view(1,-1,1,1).expand_as(self.c2_2_3.weight)
            return torch.min(post,pre)
        elif n=='c2_2_3.bias':
            return gc2_2_3.data.view(-1)
        #2-3
        elif n=='c2_3_1.weight':
            post=gc2_3_1.data.view(-1,1,1,1).expand_as(self.c2_3_1.weight)
            #pre=gct2_2.data.view(1,-1,1,1).expand_as(self.c27.weight)
            return post#return torch.min(post,pre)
        elif n=='c2_3_1.bias':
            return gc2_3_1.data.view(-1)
        elif n=='c2_3_2.weight':
            post=gc2_3_2.data.view(-1,1,1,1).expand_as(self.c2_3_2.weight)
            pre=gc2_3_1.data.view(1,-1,1,1).expand_as(self.c2_3_2.weight)
            return torch.min(post,pre)
        elif n=='c2_3_2.bias':
            return gc2_3_2.data.view(-1)
        elif n=='c2_3_3.weight':
            post=gc2_3_3.data.view(-1,1,1,1).expand_as(self.c2_3_3.weight)
            pre=gc2_3_2.data.view(1,-1,1,1).expand_as(self.c2_3_3.weight)
            return torch.min(post,pre)
        elif n=='c2_3_3.bias':
            return gc2_3_3.data.view(-1)
        #3-1
        elif n=='c3_1_1.weight':
            post=gc3_1_1.data.view(-1,1,1,1).expand_as(self.c3_1_1.weight)
            #pre=gct2_3.data.view(1,-1,1,1).expand_as(self.c37.weight)
            return post#return torch.min(post,pre)
        elif n=='c3_1_1.bias':
            return gc3_1_1.data.view(-1)
        elif n=='c3_1_2.weight':
            post=gc3_1_2.data.view(-1,1,1,1).expand_as(self.c3_1_2.weight)
            pre=gc3_1_1.data.view(1,-1,1,1).expand_as(self.c3_1_2.weight)
            return torch.min(post,pre)
        elif n=='c3_1_2.bias':
            return gc3_1_2.data.view(-1)
        elif n=='c3_1_3.weight':
            post=gc3_1_3.data.view(-1,1,1,1).expand_as(self.c3_1_3.weight)
            pre=gc3_1_2.data.view(1,-1,1,1).expand_as(self.c3_1_3.weight)
            return torch.min(post,pre)
        elif n=='c3_1_3.bias':
            return gc3_1_3.data.view(-1)
        elif n=='c3_1_d.weight':
            post=gc3_1_d.data.view(-1,1,1,1).expand_as(self.c3_1_d.weight)
            #pre=gct2_3.data.view(1,-1,1,1).expand_as(self.c45.weight)
            return post#return torch.min(post,pre)
        elif n=='c3_1_d.bias':
            return gc3_1_d.data.view(-1)
        #3-2
        elif n=='c3_2_1.weight':
            post=gc3_2_1.data.view(-1,1,1,1).expand_as(self.c3_2_1.weight)
            pre=gct3_1.data.view(1,-1,1,1).expand_as(self.c3_2_1.weight)
            return torch.min(post,pre)
        elif n=='c3_2_1.bias':
            return gc3_2_1.data.view(-1)
        elif n=='c3_2_2.weight':
            post=gc3_2_2.data.view(-1,1,1,1).expand_as(self.c3_2_2.weight)
            pre=gc3_2_1.data.view(1,-1,1,1).expand_as(self.c3_2_2.weight)
            return torch.min(post,pre)
        elif n=='c3_2_2.bias':
            return gc3_2_2.data.view(-1)
        elif n=='c3_2_3.weight':
            post=gc3_2_3.data.view(-1,1,1,1).expand_as(self.c3_2_3.weight)
            pre=gc3_2_2.data.view(1,-1,1,1).expand_as(self.c3_2_3.weight)
            return torch.min(post,pre)
        elif n=='c3_2_3.bias':
            return gc3_2_3.data.view(-1)
        #3-3
        elif n=='c3_3_1.weight':
            post=gc3_3_1.data.view(-1,1,1,1).expand_as(self.c3_3_1.weight)
            #pre=gct3_2.data.view(1,-1,1,1).expand_as(self.c59.weight)
            return post#return torch.min(post,pre)
        elif n=='c3_3_1.bias':
            return gc3_3_1.data.view(-1)
        elif n=='c3_3_2.weight':
            post=gc3_3_2.data.view(-1,1,1,1).expand_as(self.c3_3_2.weight)
            pre=gc3_3_1.data.view(1,-1,1,1).expand_as(self.c3_3_2.weight)
            return torch.min(post,pre)
        elif n=='c3_3_2.bias':
            return gc3_3_2.data.view(-1)
        elif n=='c3_3_3.weight':
            post=gc3_3_3.data.view(-1,1,1,1).expand_as(self.c3_3_3.weight)
            pre=gc3_3_2.data.view(1,-1,1,1).expand_as(self.c3_3_3.weight)
            return torch.min(post,pre)
        elif n=='c3_3_3.bias':
            return gc3_3_3.data.view(-1)
        #3-4
        elif n=='c3_4_1.weight':
            post=gc3_4_1.data.view(-1,1,1,1).expand_as(self.c3_4_1.weight)
            #pre=gct3_3.data.view(1,-1,1,1).expand_as(self.c69.weight)
            return post#return torch.min(post,pre)
        elif n=='c3_4_1.bias':
            return gc3_4_1.data.view(-1)
        elif n=='c3_4_2.weight':
            post=gc3_4_2.data.view(-1,1,1,1).expand_as(self.c3_4_2.weight)
            pre=gc3_4_1.data.view(1,-1,1,1).expand_as(self.c3_4_2.weight)
            return torch.min(post,pre)
        elif n=='c3_4_2.bias':
            return gc3_4_2.data.view(-1)
        elif n=='c3_4_3.weight':
            post=gc3_4_3.data.view(-1,1,1,1).expand_as(self.c3_4_3.weight)
            pre=gc3_4_2.data.view(1,-1,1,1).expand_as(self.c3_4_3.weight)
            return torch.min(post,pre)
        elif n=='c3_4_3.bias':
            return gc3_4_3.data.view(-1)
        #4-1
        elif n=='c4_1_1.weight':
            post=gc4_1_1.data.view(-1,1,1,1).expand_as(self.c4_1_1.weight)
            #pre=gct3_4.data.view(1,-1,1,1).expand_as(self.c79.weight)
            return post#return torch.min(post,pre)
        elif n=='c4_1_1.bias':
            return gc4_1_1.data.view(-1)
        elif n=='c4_1_2.weight':
            post=gc4_1_2.data.view(-1,1,1,1).expand_as(self.c4_1_2.weight)
            pre=gc4_1_1.data.view(1,-1,1,1).expand_as(self.c4_1_2.weight)
            return torch.min(post,pre)
        elif n=='c4_1_2.bias':
            return gc4_1_2.data.view(-1)
        elif n=='c4_1_3.weight':
            post=gc4_1_3.data.view(-1,1,1,1).expand_as(self.c4_1_3.weight)
            pre=gc4_1_2.data.view(1,-1,1,1).expand_as(self.c4_1_3.weight)
            return torch.min(post,pre)
        elif n=='c4_1_3.bias':
            return gc4_1_3.data.view(-1)
        elif n=='c4_1_d.weight':
            post=gc4_1_d.data.view(-1,1,1,1).expand_as(self.c4_1_d.weight)
            #pre=gct3_4.data.view(1,-1,1,1).expand_as(self.c87.weight)
            return post#return torch.min(post,pre)
        elif n=='c4_1_d.bias':
            return gc4_1_d.data.view(-1)
        #4-2
        elif n=='c4_2_1.weight':
            post=gc4_2_1.data.view(-1,1,1,1).expand_as(self.c4_2_1.weight)
            pre=gct4_1.data.view(1,-1,1,1).expand_as(self.c4_2_1.weight)
            return torch.min(post,pre)
        elif n=='c4_2_1.bias':
            return gc4_2_1.data.view(-1)
        elif n=='c4_2_2.weight':
            post=gc4_2_2.data.view(-1,1,1,1).expand_as(self.c4_2_2.weight)
            pre=gc4_2_1.data.view(1,-1,1,1).expand_as(self.c4_2_2.weight)
            return torch.min(post,pre)
        elif n=='c4_2_2.bias':
            return gc4_2_2.data.view(-1)
        elif n=='c4_2_3.weight':
            post=gc4_2_3.data.view(-1,1,1,1).expand_as(self.c4_2_3.weight)
            pre=gc4_2_2.data.view(1,-1,1,1).expand_as(self.c4_2_3.weight)
            return torch.min(post,pre)
        elif n=='c4_2_3.bias':
            return gc4_2_3.data.view(-1)
        #4-3
        elif n=='c4_3_1.weight':
            post=gc4_3_1.data.view(-1,1,1,1).expand_as(self.c4_3_1.weight)
            #pre=gct4_2.data.view(1,-1,1,1).expand_as(self.c101.weight)
            return post#return torch.min(post,pre)
        elif n=='c4_3_1.bias':
            return gc4_3_1.data.view(-1)
        elif n=='c4_3_2.weight':
            post=gc4_3_2.data.view(-1,1,1,1).expand_as(self.c4_3_2.weight)
            pre=gc4_3_1.data.view(1,-1,1,1).expand_as(self.c4_3_2.weight)
            return torch.min(post,pre)
        elif n=='c4_3_2.bias':
            return gc4_3_2.data.view(-1)
        elif n=='c4_3_3.weight':
            post=gc4_3_3.data.view(-1,1,1,1).expand_as(self.c4_3_3.weight)
            pre=gc4_3_2.data.view(1,-1,1,1).expand_as(self.c4_3_3.weight)
            return torch.min(post,pre)
        elif n=='c4_3_3.bias':
            return gc4_3_3.data.view(-1)
        #4-4
        elif n=='c4_4_1.weight':
            post=gc4_4_1.data.view(-1,1,1,1).expand_as(self.c4_4_1.weight)
            #pre=gct4_3.data.view(1,-1,1,1).expand_as(self.c111.weight)
            return post#return torch.min(post,pre)
        elif n=='c4_4_1.bias':
            return gc4_4_1.data.view(-1)
        elif n=='c4_4_2.weight':
            post=gc4_4_2.data.view(-1,1,1,1).expand_as(self.c4_4_2.weight)
            pre=gc4_4_1.data.view(1,-1,1,1).expand_as(self.c4_4_2.weight)
            return torch.min(post,pre)
        elif n=='c4_4_2.bias':
            return gc4_4_2.data.view(-1)
        elif n=='c4_4_3.weight':
            post=gc4_4_3.data.view(-1,1,1,1).expand_as(self.c4_4_3.weight)
            pre=gc4_4_2.data.view(1,-1,1,1).expand_as(self.c4_4_3.weight)
            return torch.min(post,pre)
        elif n=='c4_4_3.bias':
            return gc4_4_3.data.view(-1)
        #4-5
        elif n=='c4_5_1.weight':
            post=gc4_5_1.data.view(-1,1,1,1).expand_as(self.c4_5_1.weight)
            #pre=gct4_4.data.view(1,-1,1,1).expand_as(self.c121.weight)
            return post#return torch.min(post,pre)
        elif n=='c4_5_1.bias':
            return gc4_5_1.data.view(-1)
        elif n=='c4_5_2.weight':
            post=gc4_5_2.data.view(-1,1,1,1).expand_as(self.c4_5_2.weight)
            pre=gc4_5_1.data.view(1,-1,1,1).expand_as(self.c4_5_2.weight)
            return torch.min(post,pre)
        elif n=='c4_5_2.bias':
            return gc4_5_2.data.view(-1)
        elif n=='c4_5_3.weight':
            post=gc4_5_3.data.view(-1,1,1,1).expand_as(self.c4_5_3.weight)
            pre=gc4_5_2.data.view(1,-1,1,1).expand_as(self.c4_5_3.weight)
            return torch.min(post,pre)
        elif n=='c4_5_3.bias':
            return gc4_5_3.data.view(-1)
        #4-6
        elif n=='c4_6_1.weight':
            post=gc4_6_1.data.view(-1,1,1,1).expand_as(self.c4_6_1.weight)
            #pre=gct4_5.data.view(1,-1,1,1).expand_as(self.c131.weight)
            return post#return torch.min(post,pre)
        elif n=='c4_6_1.bias':
            return gc4_6_1.data.view(-1)
        elif n=='c4_6_2.weight':
            post=gc4_6_2.data.view(-1,1,1,1).expand_as(self.c4_6_2.weight)
            pre=gc4_6_1.data.view(1,-1,1,1).expand_as(self.c4_6_2.weight)
            return torch.min(post,pre)
        elif n=='c4_6_2.bias':
            return gc4_6_2.data.view(-1)
        elif n=='c4_6_3.weight':
            post=gc4_6_3.data.view(-1,1,1,1).expand_as(self.c4_6_3.weight)
            pre=gc4_6_2.data.view(1,-1,1,1).expand_as(self.c4_6_3.weight)
            return torch.min(post,pre)
        elif n=='c4_6_3.bias':
            return gc4_6_2.data.view(-1)
        #5-1
        elif n=='c5_1_1.weight':
            post=gc5_1_1.data.view(-1,1,1,1).expand_as(self.c5_1_1.weight)
            #pre=gct4_23.data.view(1,-1,1,1).expand_as(self.c311.weight)
            return post#return torch.min(post,pre)
        elif n=='c5_1_1.bias':
            return gc5_1_1.data.view(-1)
        elif n=='c5_1_2.weight':
            post=gc5_1_2.data.view(-1,1,1,1).expand_as(self.c5_1_2.weight)
            pre=gc5_1_1.data.view(1,-1,1,1).expand_as(self.c5_1_2.weight)
            return torch.min(post,pre)
        elif n=='c5_1_2.bias':
            return gc5_1_2.data.view(-1)
        elif n=='c5_1_3.weight':
            post=gc5_1_3.data.view(-1,1,1,1).expand_as(self.c5_1_3.weight)
            pre=gc5_1_2.data.view(1,-1,1,1).expand_as(self.c5_1_3.weight)
            return torch.min(post,pre)
        elif n=='c5_1_3.bias':
            return gc5_1_3.data.view(-1)
        elif n=='c5_1_d.weight':
            post=gc5_1_d.data.view(-1,1,1,1).expand_as(self.c5_1_d.weight)
            #pre=gct4_23.data.view(1,-1,1,1).expand_as(self.c319.weight)
            return post#return torch.min(post,pre)
        elif n=='c5_1_d.bias':
            return gc5_1_d.data.view(-1)
        #5-2
        elif n=='c5_2_1.weight':
            post=gc5_2_1.data.view(-1,1,1,1).expand_as(self.c5_2_1.weight)
            pre=gct5_1.data.view(1,-1,1,1).expand_as(self.c5_2_1.weight)
            return torch.min(post,pre)
        elif n=='c5_2_1.bias':
            return gc5_2_1.data.view(-1)
        elif n=='c5_2_2.weight':
            post=gc5_2_2.data.view(-1,1,1,1).expand_as(self.c5_2_2.weight)
            pre=gc5_2_1.data.view(1,-1,1,1).expand_as(self.c5_2_2.weight)
            return torch.min(post,pre)
        elif n=='c5_2_2.bias':
            return gc5_2_2.data.view(-1)
        elif n=='c5_2_3.weight':
            post=gc5_2_3.data.view(-1,1,1,1).expand_as(self.c5_2_3.weight)
            pre=gc5_2_2.data.view(1,-1,1,1).expand_as(self.c5_2_3.weight)
            return torch.min(post,pre)
        elif n=='c5_2_3.bias':
            return gc5_2_3.data.view(-1)
        #5-3
        elif n=='c5_3_1.weight':
            post=gc5_3_1.data.view(-1,1,1,1).expand_as(self.c5_3_1.weight)
            #pre=gct5_2.data.view(1,-1,1,1).expand_as(self.c333.weight)
            return post#return torch.min(post,pre)
        elif n=='c5_3_1.bias':
            return gc5_3_1.data.view(-1)
        elif n=='c5_3_2.weight':
            post=gc5_3_2.data.view(-1,1,1,1).expand_as(self.c5_3_2.weight)
            pre=gc5_3_1.data.view(1,-1,1,1).expand_as(self.c5_3_2.weight)
            return torch.min(post,pre)
        elif n=='c5_3_2.bias':
            return gc5_3_2.data.view(-1)
        elif n=='c5_3_3.weight':
            post=gc5_3_3.data.view(-1,1,1,1).expand_as(self.c5_3_3.weight)
            pre=gc5_3_2.data.view(1,-1,1,1).expand_as(self.c5_3_3.weight)
            return torch.min(post,pre)
        elif n=='c5_3_3.bias':
            return gc5_3_3.data.view(-1)
        return None