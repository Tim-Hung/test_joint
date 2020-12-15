import sys,time
import numpy as np
import torch
from torch import nn
import utils

########################################################################################################################

class Appr(object):

    def __init__(self,model,nepochs=100,sbatch=32,lr=0.1,lr_min=1e-7,lr_factor=3,lr_patience=13,clipgrad=10000,lamb=0.75,smax=400,args=None):
        self.arch ='initmodel'
        if args.resume_path:#load pretrained model if necessary
            print('loading checkpoint {}'.format(args.resume_path))
            checkpoint = torch.load(args.resume_path)
            self.arch = checkpoint['arch']

            self.begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'],strict=False)
            print('warning:last layer is 2 classes')
            model.last = nn.Linear(model.last.in_features,10)
            model.last = model.last.cuda()

            # forgetting previous task
            for n,p in model.named_parameters():
                if n.startswith('e'):
                   zero = torch.zeros_like(p.data)
                   p.data = torch.where(p.data != 0, zero, p.data)

        self.model=model

        self.nepochs=nepochs
        self.sbatch=sbatch
        self.lr=lr
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()

        self.lamb=lamb          # Grid search = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.5, 4]; chosen was 0.75
        self.smax=smax          # Grid search = [25, 50, 100, 200, 400, 800]; chosen was 400
        if len(args.parameter)>=1:
            params=args.parameter.split(',')
            print('Setting parameters to',params)
            self.lamb=float(params[0])
            self.smax=float(params[1])

        self.mask_pre=None
        self.mask_back=None
        self.args=args

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD(self.model.parameters(),lr=lr)

    def train(self,t,xtrain,ytrain,xvalid,yvalid):
        best_loss=np.inf
        best_model=utils.get_model(self.model)
        lr=self.lr
        #if int(t)<=0:lr=self.lr
        #else:lr=self.lr*0.1
        patience=self.lr_patience
        self.optimizer=self._get_optimizer(lr)

        # Loop epochs
        try:
            for e in range(self.nepochs):
                # Train
                clock0=time.time()
                self.train_epoch(t,xtrain,ytrain)
                clock1=time.time()
                train_loss,train_acc=self.eval(t,xtrain,ytrain)
                clock2=time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                    1000*self.sbatch*(clock1-clock0)/xtrain.size(0),1000*self.sbatch*(clock2-clock1)/xtrain.size(0),train_loss,100*train_acc),end='')
                # Valid
                valid_loss,valid_acc=self.eval(t,xvalid,yvalid)

                if (valid_loss<best_loss) or ((e+1)%100== 0):
                   save_file_path = self.args.output.replace('.txt','')+'t'+str(int(t))+('_{}.pth'.format(e))
                   states = {
                       'epoch': e + 1,
                       'arch': 'resnet50_hat',#arg.arch,
                       'state_dict': self.model.state_dict(),
                       'optimizer': self.optimizer.state_dict(),
                   }
                   torch.save(states, save_file_path)#save model

                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=utils.get_model(self.model)
                    patience=self.lr_patience
                    print(' *',end='')
                else:
                    patience-=1
                    if patience<=0:
                        lr/=self.lr_factor
                        print(' lr={:.1e}'.format(lr),end='')
                        if lr<self.lr_min:
                            print()
                            break
                        patience=self.lr_patience
                        self.optimizer=self._get_optimizer(lr)
                print()
        except KeyboardInterrupt:
            print()

        # Restore best validation model
        utils.set_model_(self.model,best_model)

        # Activations mask
        task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
        mask=self.model.mask(task,s=self.smax)
        for i in range(len(mask)):
            mask[i]=torch.autograd.Variable(mask[i].data.clone(),requires_grad=False)
        if t==0:
            self.mask_pre=mask
        else:
            for i in range(len(self.mask_pre)):
                self.mask_pre[i]=torch.max(self.mask_pre[i],mask[i])

        # Weights mask
        self.mask_back={}
        for n,_ in self.model.named_parameters():
            vals=self.model.get_view_for(n,self.mask_pre)
            if vals is not None:
                self.mask_back[n]=1-vals
        
        torch.save(self.model.state_dict(),'pretrain_hat.pth')
        return
    def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
    def train_epoch(self,t,x,y,thres_cosh=50,thres_emb=6):
        self.model.train()
        
        if int(t)>0:
            classname = self.model.__class__.__name__

            for module in self.model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                #if isinstance(module, nn.Linear):
                #    module.eval()
                    #print('fcmodule',module)
            #for param in self.model.parameters(): 
            #    param.requires_grad_(False)
            #    print('param#~~',n,param)
            """
            for n,p in self.model.named_parameters():
                if 'last' in n: p.requires_grad_(False)
            """

        print_model=utils.get_model(self.model)
        """print weighting for check
        for parameters in print_model:#.parameters():
            if 'last.weight' in parameters:#'bn2_1_d.running' in parameters:
            #    parameters.requires_grad_(False)
                print('parameters_last',parameters,print_model[parameters])
            if 'bn2_1_d.running' in parameters:
                print('parameters_bn',parameters,print_model[parameters])
        """
        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=torch.autograd.Variable(x[b],volatile=False)
            targets=torch.autograd.Variable(y[b],volatile=False)
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
            #s=(self.smax-1/self.smax)*i/len(r)+1/self.smax##
            s=1#20201127 delete anneal

            # Forward
            outputs,masks=self.model.forward(task,images,s=s)
            output=outputs[t]#single mask #[t]
            loss,_=self.criterion(output,targets,masks)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Restrict layer gradients in backprop
            if t>0:
                for n,p in self.model.named_parameters():
                    if n in self.mask_back:
                        p.grad.data*=self.mask_back[n]

            # Compensate embedding gradients
            for n,p in self.model.named_parameters():
                if n.startswith('e'):
                    num=torch.cosh(torch.clamp(s*p.data,-thres_cosh,thres_cosh))+1
                    den=torch.cosh(p.data)+1
                    p.grad.data*=self.smax/s*num/den##

            # Apply step
            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

            # Constrain embeddings
            for n,p in self.model.named_parameters():
                if n.startswith('e'):
                    p.data=torch.clamp(p.data,-thres_emb,thres_emb)

            #print(masks[-1].data.view(1,-1))
            #if i>=5*self.sbatch: sys.exit()
            #if i==0: print(masks[-2].data.view(1,-1),masks[-2].data.max(),masks[-2].data.min())
        #print(masks[-2].data.view(1,-1))

        return

    def eval(self,t,x,y):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        total_reg=0

        r=np.arange(x.size(0))
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=torch.autograd.Variable(x[b],volatile=True)
            targets=torch.autograd.Variable(y[b],volatile=True)
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=True)

            # Forward
            outputs,masks=self.model.forward(task,images,s=self.smax)
            output=outputs[t]#single mask #[t]
            loss,reg=self.criterion(output,targets,masks)
            #print(type(outputs))
            #print(type(output))
            _,pred=output.max(1)
            #print(_)
            #print(pred)
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.cpu().numpy().item()*len(b)
            total_acc+=hits.sum().data.cpu().numpy().item()
            total_num+=len(b)
            total_reg+=reg.data.cpu().numpy().item()*len(b)

        print('  {:.3f}  '.format(total_reg/total_num),end='')

        return total_loss/total_num,total_acc/total_num

    def criterion(self,outputs,targets,masks):
        reg=0
        count=0
        if self.mask_pre is not None:
            for m,mp in zip(masks,self.mask_pre):
                aux=1-mp
                reg+=(m*aux).sum()
                count+=aux.sum()
        else:
            for m in masks:
                reg+=m.sum()
                count+=np.prod(m.size()).item()
        reg/=count
        return self.ce(outputs,targets)+self.lamb*reg,reg

    def erase(self):
        # forgetting
        for n,p in self.model.named_parameters():
            if n.startswith('e'):
               zero = torch.zeros_like(p.data)
               p.data = torch.where(p.data < 0, zero, p.data)
    def reset_bn(self):
        #self.model.last = nn.Linear(self.model.last.in_features,10)
        #self.model.last = self.model.last.cuda()
        #bn 1-1
        #self.bn=torch.nn.BatchNorm2d(64,affine=False)
        self.model.bn1=torch.nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        #2-1
        self.model.bn2_1_1=torch.nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.model.bn2_1_2=torch.nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.model.bn2_1_3=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.model.bn2_1_d=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        #self.bnt2_1=torch.nn.Embedding(len(self.taskcla),256)
        #2-2
        self.model.bn2_2_1=torch.nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.model.bn2_2_2=torch.nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.model.bn2_2_3=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        #2-3
        self.model.bn2_3_1=torch.nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.model.bn2_3_2=torch.nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.model.bn2_3_3=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        #3-1
        self.model.bn3_1_1=torch.nn.BatchNorm2d(128, affine=False, track_running_stats=False)
        self.model.bn3_1_2=torch.nn.BatchNorm2d(128, affine=False, track_running_stats=False)
        self.model.bn3_1_3=torch.nn.BatchNorm2d(512, affine=False, track_running_stats=False)
        self.model.bn3_1_d=torch.nn.BatchNorm2d(512, affine=False, track_running_stats=False)
        #self.ect3_1=torch.nn.Embedding(len(self.taskcla),512)
        #3-2
        self.model.bn3_2_1=torch.nn.BatchNorm2d(128, affine=False, track_running_stats=False)
        self.model.bn3_2_2=torch.nn.BatchNorm2d(128, affine=False, track_running_stats=False)
        self.model.bn3_2_3=torch.nn.BatchNorm2d(512, affine=False, track_running_stats=False)
        #3-3
        self.model.bn3_3_1=torch.nn.BatchNorm2d(128, affine=False, track_running_stats=False)
        self.model.bn3_3_2=torch.nn.BatchNorm2d(128, affine=False, track_running_stats=False)
        self.model.bn3_3_3=torch.nn.BatchNorm2d(512, affine=False, track_running_stats=False)
        #3-4
        self.model.bn3_4_1=torch.nn.BatchNorm2d(128, affine=False, track_running_stats=False)
        self.model.bn3_4_2=torch.nn.BatchNorm2d(128, affine=False, track_running_stats=False)
        self.model.bn3_4_3=torch.nn.BatchNorm2d(512, affine=False, track_running_stats=False)
        #4-1
        self.model.bn4_1_1=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.model.bn4_1_2=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.model.bn4_1_3=torch.nn.BatchNorm2d(1024, affine=False, track_running_stats=False)
        self.model.bn4_1_d=torch.nn.BatchNorm2d(1024, affine=False, track_running_stats=False)
        #self.ect3_1=torch.nn.Embedding(len(self.taskcla),512)
        #4-2
        self.model.bn4_2_1=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.model.bn4_2_2=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.model.bn4_2_3=torch.nn.BatchNorm2d(1024, affine=False, track_running_stats=False)
        #4-3
        self.model.bn4_3_1=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.model.bn4_3_2=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.model.bn4_3_3=torch.nn.BatchNorm2d(1024, affine=False, track_running_stats=False)
        #4-4
        self.model.bn4_4_1=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.model.bn4_4_2=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.model.bn4_4_3=torch.nn.BatchNorm2d(1024, affine=False, track_running_stats=False)
        #4-5
        self.model.bn4_5_1=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.model.bn4_5_2=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.model.bn4_5_3=torch.nn.BatchNorm2d(1024, affine=False, track_running_stats=False)
        #4-6
        self.model.bn4_6_1=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.model.bn4_6_2=torch.nn.BatchNorm2d(256, affine=False, track_running_stats=False)
        self.model.bn4_6_3=torch.nn.BatchNorm2d(1024, affine=False, track_running_stats=False)
        #5-1
        self.model.bn5_1_1=torch.nn.BatchNorm2d(512, affine=False, track_running_stats=False)
        self.model.bn5_1_2=torch.nn.BatchNorm2d(512, affine=False, track_running_stats=False)
        self.model.bn5_1_3=torch.nn.BatchNorm2d(2048, affine=False, track_running_stats=False)
        self.model.bn5_1_d=torch.nn.BatchNorm2d(2048, affine=False, track_running_stats=False)
        #self.ect3_1=torch.nn.Embedding(len(self.taskcla),512)
        #5-2
        self.model.bn5_2_1=torch.nn.BatchNorm2d(512, affine=False, track_running_stats=False)
        self.model.bn5_2_2=torch.nn.BatchNorm2d(512, affine=False, track_running_stats=False)
        self.model.bn5_2_3=torch.nn.BatchNorm2d(2048, affine=False, track_running_stats=False)
        #5-3
        self.model.bn5_3_1=torch.nn.BatchNorm2d(512, affine=False, track_running_stats=False)
        self.model.bn5_3_2=torch.nn.BatchNorm2d(512, affine=False, track_running_stats=False)
        self.model.bn5_3_3=torch.nn.BatchNorm2d(2048, affine=False, track_running_stats=False)

########################################################################################################################
