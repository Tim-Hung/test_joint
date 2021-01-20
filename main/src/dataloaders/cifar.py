import os,sys
import numpy as np
import torch
import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle

def get(seed=0,pc_valid=0.10):
    data={}
    taskcla=[]
    size=[3,32,32]

    if not os.path.isdir('../dat/binary_cifar100/4_task'):
        os.makedirs('../dat/binary_cifar100/4_task')


        # CIFAR10
        # mean=[x/255 for x in [125.3,123.0,113.9]]
        # std=[x/255 for x in [63.0,62.1,66.7]]
        # mean=[0.4914,0.4822,0.4465]
        # std=[0.2470,0.2435,0.2616]
        # dat={}
        # dat['train']=datasets.CIFAR10('../dat/',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        # dat['train1']=datasets.CIFAR10('../dat/',train=True,download=False,transform=transforms.Compose([transforms.RandomHorizontalFlip(p=1.0),transforms.ToTensor(),transforms.Normalize(mean,std)]))
        # dat['train2']=datasets.CIFAR10('../dat/',train=True,download=False,transform=transforms.Compose([transforms.RandomRotation((-30,30)),transforms.ToTensor(),transforms.Normalize(mean,std)]))
        # dat['train3']=datasets.CIFAR10('../dat/',train=True,download=False,transform=transforms.Compose([transforms.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02), transforms.ToTensor(),transforms.Normalize(mean,std)]))
        # dat['test']=datasets.CIFAR10('../dat/',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        # for n in range(5):
        #     data[n]={}
        #     data[n]['name']='cifar10'
        #     data[n]['ncla']=2
        #     data[n]['train']={'x': [],'y': []}
        #     data[n]['test']={'x': [],'y': []}
        # for s in ['train','test']:
        #     loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        #     for image,target in loader:
        #         n=target.numpy()[0]
        #         nn=n//2
        #         data[nn][s]['x'].append(image)
        #         data[nn][s]['y'].append(n%2)

        # CIFAR100
        mean=[0.5071,0.4867,0.4408]
        std=[0.2675,0.2565,0.2761]
        dat={}
        dat['train']=datasets.CIFAR100('../dat/',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['train1']=datasets.CIFAR100('../dat/',train=True,download=True,transform=transforms.Compose([transforms.RandomHorizontalFlip(p=1.0),transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['train2']=datasets.CIFAR100('../dat/',train=True,download=True,transform=transforms.Compose([transforms.RandomRotation((-30,30)),transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['train3']=datasets.CIFAR100('../dat/',train=True,download=True,transform=transforms.Compose([transforms.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02), transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR100('../dat/',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        #for n in range(5,10):
        print('download_cifar100')
        
        predata={}
        predata['train']={'x': [],'y': []}
        predata['train1']={'x': [],'y': []}
        predata['train2']={'x': [],'y': []}
        predata['train3']={'x': [],'y': []}
        predata['test']={'x': [],'y': []}
        for s in ['train','train1','train2','train3','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
            for image,target in loader:
                n=target.numpy()[0]
                # print('n:%s'%n)
                # print('n/100:%s'%(n%100))
                predata[s]['x'].append(image)
                predata[s]['y'].append(n%100)
        
        # "Unify" and save
        for t in range(4):
            data[t]={}
            data[t]['name']='cifar100'
            data[t]['ncla']=100
            print('t:',t)
            print('ncla:',data[t]['ncla'])
            data[t]['train']={'x': [],'y': []}
            data[t]['test']={'x': [],'y': []}
            
            for s in ['train','test']:
                if t == 0 or s =='test':
                    data[t][s]['x']=torch.stack(predata[s]['x']).view(-1,size[0],size[1],size[2])
                    data[t][s]['y']=torch.LongTensor(np.array(predata[s]['y'],dtype=int)).view(-1)
                    torch.save(data[t][s]['x'], os.path.join(os.path.expanduser('../dat/binary_cifar100/4_task'),'data'+str(t)+s+'x.bin'))
                    torch.save(data[t][s]['y'], os.path.join(os.path.expanduser('../dat/binary_cifar100/4_task'),'data'+str(t)+s+'y.bin'))
                else:
                    data[t][s]['x']=torch.stack(predata[s+str(t)]['x']).view(-1,size[0],size[1],size[2])
                    data[t][s]['y']=torch.LongTensor(np.array(predata[s+str(t)]['y'],dtype=int)).view(-1)
                    torch.save(data[t][s]['x'], os.path.join(os.path.expanduser('../dat/binary_cifar100/4_task'),'data'+str(t)+s+'x.bin'))
                    torch.save(data[t][s]['y'], os.path.join(os.path.expanduser('../dat/binary_cifar100/4_task'),'data'+str(t)+s+'y.bin'))
    # Load binary files
    data={}
    #ids=list(shuffle(np.arange(10),random_state=seed))
    ids=list(np.arange(4)) # number of task
    print('Task order =',ids)
    for i in range(4):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('../dat/binary_cifar100/4_task'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('../dat/binary_cifar100/4_task'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))
        #print('i:',i)
        #print('ncla_test:',len(np.unique(data[i]['test']['y'].numpy())))
        #print('ncla_train:',data[i]['ncla'])
        if i ==0:
            data[i]['name']='cifar100-all-'+str(ids[i])
        else:
            data[i]['name']='cifar100-aug-'+str(ids[i])
        # if data[i]['ncla']==2:
        #     data[i]['name']='cifar10-'+str(ids[i])
        # else:
        #     data[i]['name']='cifar100-'+str(ids[i]-5)

    # Validation
    for t in data.keys():
        r=np.arange(data[t]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
        data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

    # Others for continual learning
#     n=0
#     for t in data.keys():
#         taskcla.append((t,data[t]['ncla']))
#         print('taskcla',taskcla)
#         n+=data[t]['ncla']
#         print('n',n)
#     data['ncla']=n
    
    # for monotonous increase
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        #print('taskcla',taskcla)
    data['ncla']=100

    return data,taskcla,size
