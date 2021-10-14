from absl import app, flags
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import pickle
from captum.attr import GradientShap
import os,sys
import time
import matplotlib.pyplot as plt
import random
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.spsa import spsa
from robustness.datasets import CIFAR

FLAGS = flags.FLAGS

# Get current working directory
cwd = os.getcwd()

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    
    
# Configuration area
#############################
batch_size = 128
num_workers = 2
#############################
use_cuda = True
device = torch.cuda.device("cuda" if use_cuda else "cpu")


################################################# Undefended CNN model ####################################################
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3,3), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3,3), padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=(3,3), padding=(1,1))
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(in_features=8*8*256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.Dropout = nn.Dropout(0.25)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.conv1(x)) #32*32*48
        x = F.relu(self.conv2(x)) #32*32*96
        x = self.pool(x) #16*16*96
        x = self.Dropout(x)
        x = F.relu(self.conv3(x)) #16*16*192
        x = F.relu(self.conv4(x)) #16*16*256
        x = self.pool(x) # 8*8*256
        x = self.Dropout(x)
        x = x.view(-1, 8*8*256) # reshape x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.Dropout(x)
        x = self.fc3(x)
        return x

############################### Adversarially-trained ResNet ##########################################################
def Load_adv_trained_model():
    
    '''train model on adversarial data'''
    
    
    
    
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
  
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    # Hard-coded dataset, architecture, batch size, workers
    ds = CIFAR('./data',num_classes=10, 
               #mean=(0.4914, 0.4822, 0.4465), 
               #std=(0.2023, 0.1994, 0.2010),
               transform_train=transform_train, transform_test=transform_test)
    #ds=cifar_trainset
    #m, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds)
    model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds,
             resume_path=os.path.join(cwd,'cifar_l2_0_5.pt'))
    model.eval()
    
    #train_loader, val_loader = ds.make_loaders(batch_size=50, workers=8)
    return model.model  

######################################## Preliminary Functions #################################################### 
def ld_cifar10(train=True):
  """Load training and test data."""
  
  if train==True:
      
      transform_train = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
      cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
      train_loader = torch.utils.data.DataLoader(cifar_trainset, batch_size=batch_size,shuffle=True, num_workers=num_workers)

  transform_test = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
  
    
  cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
  test_loader = torch.utils.data.DataLoader(cifar_testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
  
  if train==True:
      return train_loader, test_loader
  else:
      return test_loader


def data_info(test_loader):
    
    for data, target in test_loader:
        
        print('data shape: ', data.shape) #(3,32,32)
        
        break
    
def test_accuracy(net, testset_loader):
    # Test the model
    net.eval()
    correct = 0
    total = 0

    for data in testset_loader:
        images, labels = data
        #images, labels = Variable(images).cuda(), labels.cuda()
        output = net(images)
        
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy of the network is: ' + str(100 * correct / total))
    del images
    del labels
    torch.cuda.empty_cache()
    
def predict(model,img):
    
    ''' 
    predict the label of an input
    
    img : cuda float tensor (3,32,32)
    model : pytorch model
    '''

    img = img.cuda()
    img = img.reshape(1,3,32,32)
    
    output = model(img)
    _, predicted = torch.max(output.data, 1)
    del img
    torch.cuda.empty_cache()
    return predicted.item()


def prtedict_batch(model, x_batch):
    
    loader = torch.utils.data.DataLoader(x_batch,batch_size)
    
    s = 0
    for batch in loader:
      
        output = model(batch)
        if s==0:
            _, predicted = torch.max(output.data, 1)
        else:
            _, predicted_batch = torch.max(output.data, 1)
            predicted = torch.cat((predicted,predicted_batch),0)
        s+=batch_size
    
    del batch
    del predicted_batch
    torch.cuda.empty_cache()
    
    return predicted
        
def evasion_rate(model, adv_imgs,labels):
    '''Compute evasion rate of adversarial data'''
    
    model.eval()
    evaded = 0
    total = 0

    loader = torch.utils.data.DataLoader(adv_imgs,batch_size) #batch_size=50
    
    s = 0
    e = batch_size
    for batch in loader:
      
        output = model(batch)
        _, predicted = torch.max(output.data, 1)
        total += labels[s:e].size(0)
        evaded += (predicted != labels[s:e].cuda()).sum()
        s+=batch_size
        e+=batch_size
        
    del batch
    del predicted
    torch.cuda.empty_cache()
    
    return 100 * evaded / total
    
######################################################### Baseline Attacks ###############################################
def SPSA(model,test_loader,norm,size,eps,defended=False):
    
    '''perform hop_skip_jump_attack'''
    
    print('Starting spsa attack...')
    i=0
    for data in test_loader:
        x_test, y_test = data
        x_test=torch.cuda.FloatTensor(x_test.cuda())
        
        
        
        if i == 0:
            print('perturbing batch ',i+1)
            x_adv = spsa(model, x_test,eps=eps,nb_iter=10,norm = norm,sanity_checks=False)
            labels = y_test.cpu()
            torch.cuda.empty_cache()
        else:
            print('perturbing batch ',i+1)
            x_adv = torch.cat((x_adv, spsa(model, x_test,eps=eps,nb_iter=10,norm = norm,sanity_checks=False)),0)#,batch_size=batch_size
            labels = torch.cat((labels ,y_test.cpu()),0)
            torch.cuda.empty_cache()
        i+=1
        if i>0:
            print('The current evasion rate of spsa is :',evasion_rate(model, x_adv.cuda(),labels.cuda()))
            torch.cuda.empty_cache()
        if x_adv.shape[0] >= size:
            break
        #print(x_cw.shape)
        
    # save cw data
    if defended==False:
        f = open(os.path.join(cwd,'cifar','cifar-spsaL'+str(norm)), 'wb') 
    else:
        f = open(os.path.join(cwd,'cifar','defended','cifar-spsaL'+str(norm)+str(norm)), 'wb')
    pickle.dump(x_adv, f)
    f.close()
    
    return x_adv
    

def HSJ(model,test_loader,norm,size,defended=False):
    
    '''perform hop_skip_jump_attack'''
    
    print('Starting hop_skip_jump_attack attack...')
    i=0
    for data in test_loader:
        x_test, y_test = data
        x_test = x_test.cuda()
        #y_test = y_test.cuda()
        
        
        if i == 0:
            print('perturbing batch ',i+1)
            x_adv = hop_skip_jump_attack(model, x_test,batch_size=batch_size,norm = norm,verbose=0).cpu()#
            labels = y_test.cpu()
            torch.cuda.empty_cache()
        else:
            print('perturbing batch ',i+1)
            x_adv = torch.cat((x_adv, hop_skip_jump_attack(model, x_test,batch_size=batch_size, norm = norm,verbose=0).cpu()),0)#,batch_size=batch_size
            labels = torch.cat((labels ,y_test.cpu()),0)
            torch.cuda.empty_cache()
        i+=1
        if i>0:
            print('The current evasion rate of HSJ is :',evasion_rate(model, x_adv.cuda(),labels.cuda()))
            torch.cuda.empty_cache()
        if x_adv.shape[0] >= size:
            break
        #print(x_cw.shape)
        
    # save cw data
    if defended==False:
        f = open(os.path.join(cwd,'cifar','cifar-HSJL'+str(norm)), 'wb') 
    else:
        f = open(os.path.join(cwd,'cifar','defended','cifar-HSJL'+str(norm)), 'wb')
    pickle.dump(x_adv, f)
    f.close()
    
    return x_adv
    
def CW(model,test_loader,size,defended=False):
    '''Perform CW attack'''
    
    print('Starting CW attack...')
    i=0
    for data in test_loader:
        x_test, y_test = data
        x_test = x_test.cuda()
        y_test = y_test.cuda()
        
        if i == 0:
            print('perturbing batch ',i+1)
            x_cw = carlini_wagner_l2(model, x_test, 10,y_test,targeted = False)
            labels = y_test
        else:
            print('perturbing batch ',i+1)
            x_cw = torch.cat((x_cw, carlini_wagner_l2(model, x_test, 10,y_test,targeted = False)),0)
            labels = torch.cat((labels ,y_test),0)
        i+=1
        print(x_cw.shape)
        #if i%10!=0:
        print('The current evasion rate of CW is :',evasion_rate(model, x_cw,labels))
        if x_cw.shape[0]>=size:
            break
        #print(x_cw.shape)
        
    # save cw data
    #f = open(os.path.join(cwd,'cifar','cifar-CW'+str(x_cw.shape[0])), 'wb') 
    
    if defended==False:
        f = open(os.path.join(cwd,'cifar','cifar-CW'+str(x_cw.shape[0])), 'wb') 
    else:
        f = open(os.path.join(cwd,'cifar','defended','cifar-CW'+str(x_cw.shape[0])), 'wb')
    pickle.dump(x_cw, f)
    f.close()
    return x_cw
   
def FGS(model,test_loader,norm,size,eps,defended=False):
    '''Perform FGS attack'''
    
    
    print('Starting FGS attack...')
    i=0
    
    for data in test_loader:
        x_test, y_test = data
        
        x_test=x_test.cuda()
        #y_test=y_test.cuda()
        
        if i == 0:
            x_fgm = fast_gradient_method(model, x_test,eps=eps, norm=norm)
            labels = y_test
        else:
            x_fgm = torch.cat((x_fgm, fast_gradient_method(model, x_test, eps = eps, norm = norm)),0)
            labels = torch.cat((labels ,y_test),0)
        i+=1
        
        print(x_fgm.shape)
        print('The current evasion rate of FGS is :',evasion_rate(model, x_fgm,labels.cuda()))
        if x_fgm.shape[0]>=size:
            break
        
        
    # save fgm data
    del labels
    del x_test
    torch.cuda.empty_cache()
    
    if defended==False:
        f = open(os.path.join(cwd,'cifar','cifar-FGSL'+str(norm)+str(eps)), 'wb') 
    else:
        f = open(os.path.join(cwd,'cifar','defended','cifar-FGSL'+str(norm)+str(eps)), 'wb')
    
    pickle.dump(x_fgm, f)
    f.close()
    
    return x_fgm
    
def PGD(model,test_loader,norm,size,eps,defended=False):
    '''Perform PGD attack'''
    
    print('Starting PGD attack...')
    i=0
    for data in test_loader:
        x_test, y_test = data
        
        x_test = x_test.cuda()
        #y_test = y_test.cuda()
        if i == 0:
            x_pgd = projected_gradient_descent(model, x_test, eps, 0.01, 40, norm)
            labels = y_test
        else:
            x_pgd = torch.cat((x_pgd, projected_gradient_descent(model, x_test, eps, 0.01, 40, norm)),0)
            labels = torch.cat((labels ,y_test),0)
        i+=1
        
        print(x_pgd.shape)
        print('The current evasion rate of PGD is :',evasion_rate(model, x_pgd,labels.cuda()))
        
        if x_pgd.shape[0]>=size:
            break
        
        
    del labels
    del x_test
    torch.cuda.empty_cache()
        
    # save PGD data
    
    if defended==False:
        f = open(os.path.join(cwd,'cifar','cifar-PGDL'+str(norm)+str(eps)), 'wb') 
    else:
        f = open(os.path.join(cwd,'cifar','defended','cifar-PGDL'+str(norm)+str(eps)), 'wb')
    pickle.dump(x_pgd, f)
    f.close()
    
    return x_pgd

####################################################### EG-Booster Functions ##################################################
def deepexplain(test_loader,path,size, model):
    
    # perform shap explanations on original images
    
    gradient_shap = GradientShap(model)
    
    shp =[]
    print('Starting SHAP Gradientexplainer...')
    start_time = time.time()
    j=0
    for data in test_loader:
        org_img, labels = data
        del data
        torch.cuda.empty_cache()
        org_img = torch.cuda.FloatTensor(org_img.cuda())
        #print(org_img.shape)
        print('Explainig batch {}'.format(j+1))
        for i in range(labels.shape[0]):
            rand_img_dist = torch.cat([org_img[i:i+1] * 0, org_img[i:i+1] * 255])
            attributions_gs = gradient_shap.attribute(org_img[i:i+1],
                                          n_samples=100,
                                          stdevs=0.0001,
                                          baselines=rand_img_dist,
                                          target=labels[i].item())
            
            shp_i = attributions_gs.squeeze().cpu().detach().numpy()
            shp.append(shp_i)
        if j>=int(size/batch_size):
            break
        j+=1
    
    print((time.time() - start_time)/3600,' hrs')
    shp = np.array(shp)
    np.save(path, shp)
    torch.cuda.empty_cache()
    
    return shp



def One_EG_Attack(org_img,adv_img,shaply,label,model,eps,norm_type):
    
    '''
    org_img : original image (3,32,32)
    adv_img : adv image (3,32,32)
    shaply: shapley values of the true label (3,32,32)
    label : the true label of org_img
    model : pytorch model
    '''
    
    lines = 32
    cols = 32
    channels = 3
    ege_best = adv_img.detach().clone()
    pos_unperturbed = 0
    tot = 0
    pert_change = 0
    #std_red = 0.2023
    #std_green = 0.1994
    #std_blue = 0.2010
    out_bound = 0
    attmpt = 0
    
    norm = round(torch.norm((adv_img.cpu() - org_img.cpu()).cuda(), norm_type).item(),2)
    #print('Initial perturbation size:',norm)
    
    if norm > eps:
        raise ValueError("norm {} cannot be used : Choose the same norm order used for initial attack".format(norm_type))
    
    # eliminating non-consequential perturbation
    for i in range(channels):
        for j in range(cols):
            for k in range(lines):
                
                if adv_img[i,j,k].item() != org_img[i,j,k].item():
                    tot+=1
                    
                    if shaply[i,j,k] < 0:
                        
                        # if pixel is originally negative to the original prediction but is perturbed
                        ege_best[i,j,k] = org_img[i,j,k].item()
                        if predict(model,adv_img) != label and predict(model,ege_best) == label:
                            ege_best[i,j,k] = adv_img[i,j,k].item()
                        else:
                            pert_change-=1
    
    # Adding positive (consequential) perturbations
    if predict(model,ege_best) == label:
        
        for i in range(channels):
            for j in range(cols):
                for k in range(lines):
                            
                    if shaply[i,j,k] >= 0:
                        attmpt+=1
                        
                        if adv_img[i,j,k].item() == org_img[i,j,k].item():
                            pos_unperturbed+=1
                            # if pixel is originally positive to the original prediction but not perturbed
                        
                        if predict(model,ege_best) == label:
                            pert = random.random()
                            ege_best[i,j,k] += pert
                            
                            # Checking perturbation bound
                            
                            attempt = 0
                            reduce = 1
                            while attempt < 20 and round(torch.norm((ege_best.cpu()- org_img.cpu()).cuda(), norm_type).item(),2) > eps:
                                reduce +=1
                                pert = pert/reduce
                                ege_best[i,j,k] = adv_img[i,j,k] + pert
                                attempt += 2
                            
                            if round(torch.norm((ege_best.cpu() - org_img.cpu()).cuda(), norm_type).item(),2) > eps:
                                ege_best[i,j,k] = adv_img[i,j,k]
                                out_bound+=1
                            
                        else:
                            break
    
    
    if attmpt != 0:
        out_bound_rate = out_bound/attmpt
    else:
        out_bound_rate = 0
    if tot !=0:
        change_rate = pert_change/float(tot)
    else:
        change_rate = 0
    return ege_best,change_rate  , pos_unperturbed,out_bound_rate
    
    
def EG_Attack(start,test_loader,adv_loader,model,shp,attack_name,eps, norm_type, run,size,defended=False):
    
    ''' Perform giuided attack over all adv data
    '''
    
    if norm_type not in [np.inf, 1, 2]:
        raise ValueError(
            "Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm_type)
        )
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    out_boundAll = []
    # Get original data
    i=0
    start_time = time.time()
    max_change = 0
    tot_change = []
    tot_pos =0
    
    for data,adv_imgs  in zip(test_loader,adv_loader):
        x_batch, y_batch = data
        if i==0:
            labels = y_batch
        else:
            labels = torch.cat((labels, y_batch))
        
        print('Performing EG-Booster on batch {} ...'.format(i+1))
        for ind in range(batch_size):
            
            new_adv , pert_change, pos_unpert, out_bound = One_EG_Attack(x_batch[ind],adv_imgs[ind],shp[ind+i*batch_size],y_batch[ind].item(),model,eps,norm_type)
            
            
            if abs(pert_change) > max_change:
                max_change = pert_change
            tot_change.append(pert_change)
            tot_pos += pos_unpert
            out_boundAll.append(out_bound)
            if ind == 0 and i==0:
                best_adv=new_adv.reshape((1,3,32,32))
            else:
                best_adv = torch.cat((best_adv,new_adv.reshape((1,3,32,32))),0)
            
        print('Processed shape:',best_adv.shape)  
        print('Current EG-Booster Evasion rate on {} is {} %'.format(attack_name,evasion_rate(model, torch.cuda.FloatTensor(best_adv.cuda()),labels.cuda())))      
        if labels.shape[0] == size:
            break
        i+=1
    
    
    print((time.time() - start_time)/3600,' hrs')
    # save new adv
    '''
    if defended==False:
        f = open(os.path.join(cwd,'cifar','EG-'+attack_name+str(norm_type)+'_'+str(eps)), 'wb') 
    else:
        f = open(os.path.join(cwd,'cifar','defended','EG-'+attack_name+str(norm_type)+'_'+str(eps)), 'wb') 
    pickle.dump(best_adv, f)
    f.close()
    '''
    
    print('Evasion rate of EG-Booster attack is {} %'.format(evasion_rate(model, torch.cuda.FloatTensor(best_adv.cuda()),labels.cuda())))
    
    print('Max change :{}%'.format(max_change*100))
    print('Avg change :{}%'.format(np.mean(tot_change)*100))
    print('Avg unperturbed positives :{}%'.format(float(tot_pos)/labels.shape[0]*100))
    print('Avg number of outbound EG-Booster perturbations {}%'.format(np.mean(out_boundAll)*100))
    
    return best_adv


def similarity(run1,run2):
    
    if run1.shape[0] != run2.shape[0]:
        raise ValueError("run1 shape {} is different than run2 shape {}".format(run1.shape,run2.shape))
    
    size = 0
    for i in range(run1.shape[0]):
        if run1[i] == run2[i]:
            i+=1
    
    sim = (run1*run2>=0).sum()/np.prod(list(run1.shape))   
    
    return sim

def similarity_shp(shp1,shp2,l,x_len):
    
    '''Compute similarities of SHAP explanations'''
    sim = []
    for ind in range(x_len): # across different samples
        
        shp1_c=shp1[ind].reshape((np.prod(list(shp1[ind].shape))))
        shp2_c=shp2[ind].reshape((np.prod(list(shp2[ind].shape))))
        #print(list(shp1_c))
        #print(list(shp2_c))
        #print(shp1_c.shape,shp2_c.shape)
        argmax1=[]
        argmax2=[]
        for i in range(l):
            
            # Update argmax1
            argmax11=np.argmax(shp1_c)
            argmax1.append(argmax11)
            # Update argmax2
            argmax22=np.argmax(shp2_c)
            argmax2.append(argmax22)
            # Next argmax
            if shp2_c[argmax22] == 0 or shp1_c[argmax11] == 0:
                break
            else:
                shp2_c[argmax22] = np.min(shp2_c)-1
                shp1_c[argmax11] = np.min(shp1_c)-1
        
        inter = list(set(argmax1) & set(argmax2))
        sim.append(2* len(inter)/(len(argmax1) + len(argmax2)))
        
    return np.mean(sim)

def stability(k,test_loader,l,attack,norm,eps,model,size,runs):
    '''Compute the stability of EG-Booster'''
    
    
    if len(runs) != k:
        raise ValueError('Given list of runs length ({}) should be equal to k ({})'.format(len(runs),k))
    
    
    stb_shp=[]
    stb_eg=[]
    for i in range(k):
        s_eg=[]
        s_shp=[]
        for j in range(k):
            if i!=j:
                print('Computing similarities of run {} with run {}'.format(i,j))
                base_run=np.load(os.path.join(cwd,'cifar','shp'+str(i)+'.npy'))
                temp_run=np.load(os.path.join(cwd,'cifar','shp'+str(j)+'.npy'))
                s_eg.append(similarity(runs[i],runs[j]))
                
                s_shp.append(similarity_shp(base_run,temp_run,l,size))
        print(s_eg)    
        print('Current similarity of shap: ',np.mean(s_shp))
        print('Current similarity of shap: ',np.mean(s_eg))
        stb_shp.append(np.mean(s_shp))
        stb_eg.append(np.mean(s_eg))
                
    return np.mean(stb_eg), np.mean(stb_shp)
    
def stability_plot(k,test_loader,l,attack,norm,eps,model,size,path,defended=False):
    
    runs = []
    
    # Load results of baseline attack
    f = open(os.path.join(cwd,'cifar','cifar-'+attack+'L'+str(norm)+str(eps)), 'rb')
    X_adv = pickle.load(f)
    f.close()
    
    # Get baseline adv data
    loader = torch.utils.data.DataLoader(X_adv,batch_size)
    del X_adv
    torch.cuda.empty_cache()
    
    # run EG-Booster k times
    for i in range(k):
        print('Run {} of EG-booster: '.format(i))
        shp=np.load(os.path.join(cwd,'cifar','shp'+str(i)+'.npy'))
        ege_best=EG_Attack(0,test_loader,loader,model,shp,attack,eps, norm,1,size,defended=False)
        runs.append(np.array(prtedict_batch(model, ege_best).cpu()))
      
    print(runs)
        
    shp_stab=[]
    eg_stab=[]
    
    for k_i in range(k):
        if k_i in [0,1]:
            shp_stab.append(1.0)
            eg_stab.append(1.0)
        else:
            stb_eg_k, stb_shp_k = stability(k_i,test_loader,l,attack,norm,eps,model,size,runs[:k_i])
            shp_stab.append(stb_shp_k)
            eg_stab.append(stb_eg_k)
     
    print(shp_stab)
    print(eg_stab)
    
    x=range(k)
    plt.plot(x, shp_stab,color='red',label='SHAP')
    plt.plot(x, eg_stab,color='blue',label='EG-Booster')
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("stability")
    plt.savefig(path)
    
   
def eps_plot(test_loader,size,model,path,attack,defended=False):
    
    
    # Load shap explanations
    if defended == False:
        shp = torch.cuda.FloatTensor(np.load(os.path.join(cwd,'cifar','shp0.npy')))
    else:
        shp = torch.cuda.FloatTensor(np.load(os.path.join(cwd,'cifar','defended','shp0.npy')))
    shp = shp[:size]
    # Load correct labels
    i=0
    for data in test_loader:
        _, y_test = data
        #x_test = x_test.cuda()
        
        if i == 0:
            labels = y_test
        else:
            labels = torch.cat((labels,y_test),0)   
        i+=1
        if labels.shape[0]>=size:
            break
        del y_test
        torch.cuda.empty_cache()
        
    epsilons = [0.1,0.2,0.4,0.6,0.7]
    base_rate=[]
    booster_rate=[]
    for eps in epsilons:
        #flags.DEFINE_float("eps", eps, "Total epsilon for FGM and PGD attacks.")
        print('####eps=',eps)
        print('###Performing baseline attack...')
        if defended==False:
            if attack == 'PGD':
                adv_imgs = PGD(model,test_loader,2,size,eps,defended=False)
                    
            elif attack == 'FGS':
                adv_imgs = FGS(model,test_loader,2,size,eps,defended=False)
                
            else:
                raise ValueError("only FGS and PGD are currrently supported for CIFAR10-CNN different epsilon analysis")
        else:
            if attack == "spsa":
                adv_imgs = SPSA(model,test_loader,np.inf,size,eps,defended=True)
            else:
                raise ValueError("Only spsa attack is currrently supported for ResNet50 different epsilon analysis")
            
        adv_imgs = adv_imgs[:size]
        print(adv_imgs.shape)
                
        base_rate.append(evasion_rate(model, adv_imgs,labels).item())
        print('###Performing EG-Booster attack...')
        loader = torch.utils.data.DataLoader(adv_imgs,batch_size)
        del adv_imgs 
        torch.cuda.empty_cache()
        
        if defended==False:
            ege_best = EG_Attack(0,test_loader,loader,model,shp,attack+str(eps),eps, 2,1,size,defended=False)
        else:
            ege_best = EG_Attack(0,test_loader,loader,model,shp,attack+str(eps),eps, np.inf,1,size,defended=True)
        print(ege_best.shape)
        booster_rate.append(evasion_rate(model, ege_best,labels).item())
    
    print(base_rate)
    print(booster_rate)
    x=epsilons
    plt.plot(x, base_rate,linestyle='--', marker='o', color='red',label=attack)
    plt.plot(x, booster_rate,linestyle='--', marker='o', color='blue',label='EG-'+attack)
    plt.legend()
    plt.xlabel("epsilon")
    plt.ylabel("Evasion Rate")
    plt.grid()
    plt.savefig(path)  

def show_img(img, path):
    
    fig = plt.figure
    img = img / 2 + 0.5   # unnormalize
    npimg = img.cpu().detach().numpy()   # convert from tensor
    plt.imshow(np.transpose(npimg, (1, 2, 0))) 
    
    '''
    fig, axes1 = plt.subplots(5,5,figsize=(3,3))
    for j in range(5):
        for k in range(5):
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(img)
    '''
    plt.savefig(path)
        
def main(_):
  
  torch.cuda.empty_cache()
  print('Loading data...')
  test_loader = ld_cifar10(train=False)
  
  # Comment the following if the defended model is used
  print('Loading model...')
  model = ConvNet()
  model.cuda()
  model.load_state_dict(torch.load(os.path.join(cwd,'cifar',"model_with_epoch" + str(50) + ".pth")))
  
  # Comment the following if the Undefended model is used
  '''
  print('Loading defended model...')
  model=Load_adv_trained_model()
  
  '''
  
  
  # Run SHAP explanations
  if FLAGS.defended == True:
      shp=deepexplain(test_loader,os.path.join(cwd,'cifar','defended','shp'+str(0)),FLAGS.size, model)
  else:
      shp=deepexplain(test_loader,os.path.join(cwd,'cifar','shp'+str(0)),FLAGS.size, model)
  
  # Perform FGSL2 attack
  x_fgm = FGS(model,test_loader,FLAGS.norm,FLAGS.size,FLAGS.eps,defended=FLAGS.defended)
  
  
  # Create a loader for baseline attack data
  loader = torch.utils.data.DataLoader(x_fgm,batch_size)
  del x_fgm
  torch.cuda.empty_cache()
  
  # Perform EG-Booster attack
  print('\n Performing EG-Booster...')
  best_adv = EG_Attack(0,test_loader,loader,model,shp,'FGS',FLAGS.eps, FLAGS.norm,1,FLAGS.size,defended=FLAGS.defended)
  
  # Perfom 10 runs of SHAP explanations for stability analysis
  print('\n Performing Stability Analysis...')
  if FLAGS.defended == False:
      for i in range(1,10): 
          shp=deepexplain(test_loader,os.path.join(cwd,'cifar','shp'+str(i)),1000, model)
          print(shp.shape)
  
      # Perform Stability Analysis
      stability_plot(10,test_loader,10,'FGS',FLAGS.norm,FLAGS.eps,model,1000,os.path.join(cwd,'cifar','stability'+'_FGSL'+str(FLAGS.norm)+'.png'))
      
      # Plot evasion rate curve for different epsilons
      eps_plot(test_loader,5000,model,os.path.join(cwd,'cifar','eps_anaFGS.png'),'FGS',defended=False)
  
  
#### Change the following config as suited
if __name__ == '__main__':
    flags.DEFINE_integer("size", 10000, "Test set size")
    flags.DEFINE_float("eps", 0.3, "Total epsilon for FGS and PGD attacks.")
    flags.DEFINE_integer("norm", 2, "Used distance metric.")
    flags.DEFINE_bool(
        "defended", False, "Use the undefended model"
    )

    app.run(main)
