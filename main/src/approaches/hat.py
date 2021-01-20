import sys,time
import numpy as np
import torch
import torch.nn.functional as F
import utils
import datetime
date = datetime.datetime.now()

""
class Appr(object):

    def __init__(self,model,nepochs=100,sbatch=100,lr=0.01,lr_min=0,lr_factor=3,lr_patience=4,clipgrad=10000,lamb=0.75,smax=400,args=None):
        self.model=model
        self.optim='SGD_momentum'
        self.nepochs=nepochs
        self.sbatch=sbatch
        self.lr=lr
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad
        self.lr_scheduler='Original'#'MultiStepLR'#'ReduceLROnPlateau'
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
        
        return

    def _get_optimizer(self,lr=None,optimizer='SGD_momentum'):
        if lr is None: lr=self.lr
        if optimizer is 'SGD': 
            optimizer = torch.optim.SGD(self.model.parameters(),lr=lr)
            print('optimizer is SGD')
        if optimizer is 'SGD_momentum':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) 
            print('optimizer is SGD_momentum')
        if optimizer is 'Adam':optimizer=torch.optim.Adam(self.model.parameters(),lr=lr)
        return optimizer

    def train(self,t,xtrain,ytrain,xvalid,yvalid,args):
        # Loop epochs
        # if t == 0:
        #     self.nepochs = 60
        # else:
        #     self.nepochs = 60
            
        best_loss=np.inf
        best_val_acc=0
        best_model=utils.get_model(self.model)
        lr=self.lr
        optimizer=self.optim
        patience=self.lr_patience
        self.optimizer=self._get_optimizer(lr, optimizer)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.05, patience=3,
 verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        miles = []
        for i in range(3):
            miles.append(int(self.nepochs*0.8**(i+1)))
        print(sorted(miles))
        list_file = open(args.output, 'a')
        list_file.write('milestones:%s'%(sorted(miles),)+'\n')
        list_file.close
        train_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=sorted(miles), gamma=0.2) #learning rate 
        
            
        print('%s%s%s%s%s%s_%s_%s_lr%s_factor%s_task%s'%(date.year,date.month,date.day,date.hour,date.minute,date.second,self.lr_scheduler,self.optim,self.lr,self.lr_factor,t))
        print('epochs:%s'%self.nepochs)
        list_file = open(args.output, 'a')
        list_file.write('%s%s%s%s%s%s_%s_%s_lr%s_factor%s_task%s'%(date.year,date.month,date.day,date.hour,date.minute,date.second,self.lr_scheduler,self.optim,self.lr,self.lr_factor,t)+'\n')
        list_file.close
        try:
            for e in range(self.nepochs):
                # Train
                clock0=time.time()
                self.train_epoch(t,xtrain,ytrain)
                clock1=time.time()
                train_loss,train_acc,_,__,___=self.eval(t,xtrain,ytrain,args)
                clock2=time.time()
                list_file = open(args.output, 'a')
                list_file.write('| Epoch:%s, time=%.4f ms/ %.4f ms | Train: loss=%.4f, acc=%.4f %%|'%(e+1,1000*self.sbatch*(clock1-clock0)/xtrain.size(0),1000*self.sbatch*(clock2-clock1)/xtrain.size(0),train_loss,100*train_acc))
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                    1000*self.sbatch*(clock1-clock0)/xtrain.size(0),1000*self.sbatch*(clock2-clock1)/xtrain.size(0),train_loss,100*train_acc),end='')
                # Valid
                valid_loss,valid_acc,_,__,___=self.eval(t,xvalid,yvalid,args)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
                list_file.write(' Valid: loss=%.4f, acc=%.4f %%|'%(valid_loss,100*valid_acc) + '\n')
                list_file.close
                if self.lr_scheduler == 'ReduceLROnPlateau':
                # ReduceLROnPlateau
                    scheduler.step(train_loss)
                elif self.lr_scheduler == 'MultiStepLR' :     
                    train_scheduler.step(e)
                else:
                # Adapt lr
                    if train_loss<best_loss:
                        best_loss=train_loss
#                         best_epoch=e
#                         best_model=utils.get_model(self.model)
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
                if valid_acc>best_val_acc :
                    best_val_acc=valid_acc
                    best_epoch=e
                    best_model=utils.get_model(self.model)
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
        torch.save(best_model,'../model/%s%s%s%s%s%s_%s_%s_lr%s_factor%s_BestEpoch_%s_task%s.pth'%(date.year,date.month,date.day,date.hour,date.minute,date.second,self.lr_scheduler,self.optim,self.lr,self.lr_factor,best_epoch,t))
        list_file = open(args.output, 'a')
        list_file.write('Model saved at ../model/%s%s%s%s%s%s_%s_%s_lr%s_factor%s_BestEpoch_%s_task%s.pth'%(date.year,date.month,date.day,date.hour,date.minute,date.second,self.lr_scheduler,self.optim,self.lr,self.lr_factor,best_epoch,t) + '\n')
        list_file.close   
        
        return

    def train_epoch(self,t,x,y,thres_cosh=50,thres_emb=6):
        self.model.train()
        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).cuda()
#         print(len(r))
#         print(self.sbatch)
        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): 
                b=r[i:i+self.sbatch]
            else:
                b=r[i:]
            images=torch.autograd.Variable(x[b],volatile=False)
            targets=torch.autograd.Variable(y[b],volatile=False)
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
            s=(self.smax-1/self.smax)*i/len(r)+1/self.smax
            # Forward
            outputs,masks=self.model.forward(task,images,s=s)
            output=outputs[t]
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
                    p.grad.data*=self.smax/s*num/den

            # Apply step
            #torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
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

    def eval(self,t,x,y,args):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        total_reg=0

        r=np.arange(x.size(0))
        r=torch.LongTensor(r).cuda()
#         total_pred=torch.FloatTensor(2,20).fill_(0)
        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=torch.autograd.Variable(x[b],volatile=True)
            targets=torch.autograd.Variable(y[b],volatile=True)
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=True)

            # Forward
            outputs,masks=self.model.forward(task,images,s=self.smax)
            output=outputs[t]
#             print(output.shape)
            loss,reg=self.criterion(output,targets,masks)
            
            prob,pred=output.max(1)
            
            # normalization
            _range = torch.max(prob) - torch.min(prob)
            prob = (prob - torch.min(prob)) / _range

#             print(prob.shape)
#             print(pred.shape)
#             # list_pred=torch.stack((prob,pred.float()),0)
#             if i == 0:
#                 list_prob = prob
#                 list_pred = pred
#                 list_targets = targets
#             else:
#                 list_prob=torch.cat((list_prob,prob),0)
#                 list_pred=torch.cat((list_pred,pred),0)
#                 list_targets=torch.cat((list_targets,targets),0)
#             print(list_prob.shape)
#             print(list_pred.shape)
            if i == 0:
                list_prob = []
                list_pred = []
                list_targets = []
                list_prob.append(prob.cpu().detach().numpy().squeeze().tolist())
                list_pred.append(pred.cpu().detach().numpy().squeeze().tolist())
                list_targets.append(targets.cpu().detach().numpy().squeeze().tolist())
                cnt = 1
                total = len(prob.cpu().detach().numpy().squeeze().tolist())
                #print("total:",total)
            else:
                list_prob.append(prob.cpu().detach().numpy().squeeze().tolist())
                list_pred.append(pred.cpu().detach().numpy().squeeze().tolist())
                list_targets.append(targets.cpu().detach().numpy().squeeze().tolist())
                a = len(prob.cpu().detach().numpy().squeeze().tolist())
                cnt += 1
                total = total + a
                #print("total:",total)
            #print('cnt', cnt)
            hits=(pred==targets).float()
            #print('hits shape:', hits.shape)
            #print('hits:', hits)

            # Log
            total_loss+=loss.data.cpu().numpy().item()*len(b)
            total_acc+=hits.sum().data.cpu().numpy().item()
            total_num+=len(b)
            total_reg+=reg.data.cpu().numpy().item()*len(b)
            
        #print('  {:.3f}  '.format(total_reg/total_num),end='')
#         print('prob:%s'%len(list_prob))
#         print('pred:%s'%len(list_pred))
#         print('targets:%s'%len(list_targets))
#         return total_loss/total_num,total_acc/total_num,list_prob.cpu().detach().numpy().squeeze(),list_pred.cpu().detach().numpy().squeeze(),list_targets.cpu().detach().numpy().squeeze().tolist()
        return total_loss/total_num,total_acc/total_num,list_prob,list_pred,list_targets
    
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

# #######################################################################################################################
#             if i == 0:
#                 list_prob = prob.cpu().detach().numpy().squeeze()
#                 list_pred = pred.cpu().detach().numpy().squeeze()
#                 list_targets = targets.cpu().detach().numpy().squeeze().tolist()
#             else:
#                 list_prob=torch.cat((list_prob,prob.cpu().detach().numpy().squeeze()),0)
#                 list_pred=torch.cat((list_pred,pred.cpu().detach().numpy().squeeze()),0)
#                 list_targets=torch.cat((list_targets,targets.cpu().detach().numpy().squeeze().tolist()),0)
