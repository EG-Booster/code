# -*- coding: utf-8 -*-

from absl import app, flags
import numpy as np
import os, sys
import pickle
import time
import shap
import torch
from c_attack_evasion_cleverhans import CAttackEvasionCleverhans
'''
from cleverhans.torch.attacks import \
    FastGradientMethod, CarliniWagnerL2, \
    ProjectedGradientDescent, MomentumIterativeMethod, \
    BasicIterativeMethod
'''
from cleverhans.torch.attacks.fast_gradient_method import FastGradientMethod
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.basic_iterative_method import BasicIterativeMethod
from cleverhans.torch.attacks.carlini_wagner_l2 import CarliniWagnerL2
from cleverhans.torch.attacks.momentum_iterative_method import MomentumIterativeMethod

from secml.data.c_dataset import CDataset
from secml.data.loader import CDataLoaderMNIST
from secml.model_zoo import load_model
from secml.ml.peval.metrics import CMetric
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


FLAGS = flags.FLAGS
# Get current working directory
cwd = os.getcwd()

# Create the SecML accuracy Metric
metric = CMetric.create('accuracy')


###############################################  Baseline Attacks   ####################################### 

def FGS(clf,attack_X,attack_y, norm,eps):
    
    ''' Perform Fast-Gradient sign Attack
    '''
    

    FGS_attack = CAttackEvasionCleverhans(clf, y_target=None, clvh_attack_class=FastGradientMethod, store_var_list=None,eps=eps, norm = norm)

    print("Attack started...")
    with suppress_stdout():
        start_time = time.time()
        eva_y_pred, _, eva_adv_ds, _ = FGS_attack.run(attack_X, attack_y)
    print((time.time() - start_time)/3600,' hrs')
    print("Attack complete!")
    
    acc = metric.performance_score(
        y_true=attack_y, y_pred=clf.predict(attack_X))
    acc_attack = metric.performance_score(
        y_true=attack_y, y_pred=clf.predict(eva_adv_ds.X))
    
    print("Accuracy on test set before attack: {:.2%}".format(acc))
    print("Accuracy on test set after FGS norm {} attack: {:.2%}".format(norm,acc_attack))
    
    f = open(os.path.join(cwd,'FGSL'+str(norm)), 'wb') 
    pickle.dump(eva_adv_ds, f)
    
    
    return eva_adv_ds.X


def BIM(clf,attack_X,attack_y,norm,eps):
    
    ''' Perform Basic Iterative Method Attack
    '''
    
    
    if norm == np.inf:
        BIM_attack = CAttackEvasionCleverhans(clf, y_target=None, clvh_attack_class=BasicIterativeMethod, store_var_list=None,eps=eps,norm=norm,
                 clip_min= 0.,
                 clip_max= 1.)
    else:
        raise ValueError("norm {} is not supported: use inf".format(norm))
    
    
    
    
    print("Attack started...")
    start_time = time.time()
    eva_y_pred, _, eva_adv_ds, _ = BIM_attack.run(attack_X, attack_y)
    print((time.time() - start_time)/3600,' hrs')
    print("Attack complete!")
    
    acc = metric.performance_score(
        y_true=attack_y, y_pred=clf.predict(attack_X))
    acc_attack = metric.performance_score(
        y_true=attack_y, y_pred=clf.predict(eva_adv_ds.X))
    
    print("Accuracy on reduced test set before attack: {:.2%}".format(acc))
    print("Accuracy on reduced test set after BIM norm {} attack: {:.2%}".format(norm,acc_attack))
    
    f = open(os.path.join(cwd,'BIML'+str(norm)), 'wb') 
    pickle.dump(eva_adv_ds, f)
    
    
    return eva_adv_ds.X

def CW(clf,attack_X,attack_y,eps):
    
    ''' Perform Carlini&Wagner Method Attack
    '''
    
    attack = CAttackEvasionCleverhans(clf, y_target=None, clvh_attack_class=CarliniWagnerL2, store_var_list=None,eps=eps)
    
    print("Attack started...")
    start_time = time.time()
    eva_y_pred, _, eva_adv_ds, _ = attack.run(attack_X, attack_y)
    print((time.time() - start_time)/3600,' hrs')
    print("Attack complete!")
    
    f = open(os.path.join(cwd,'CW'+str(100)), 'wb')
    pickle.dump(eva_adv_ds, f)
    
    acc = metric.performance_score(
        y_true=attack_y, y_pred=clf.predict(attack_X))
    acc_attack = metric.performance_score(
        y_true=attack_y, y_pred=clf.predict(eva_adv_ds.X))
    
    print("Accuracy on test set before attack: {:.2%}".format(acc))
    print("Accuracy on test set after attack: {:.2%}".format(acc_attack))
    
    
    
    
    return eva_adv_ds.X

def pgd(clf,attack_X,attack_y,tr,norm,eps):
    
    ''' Perform projected-gradient method attack
    '''
    
    if norm == 2:
        noise_type = 'l2'  # Type of perturbation 'l1' or 'l2'
    elif norm == 1:
        noise_type = 'l1'  # Type of perturbation 'l1' or 'l2'
    elif norm == np.inf:
        pgd_ls_attack = CAttackEvasionCleverhans(clf, y_target=None, clvh_attack_class=ProjectedGradientDescent, store_var_list=None,norm=norm,eps=eps)
    else:
        raise ValueError("Norm {} is not implemented for PGD attack".format(norm))
    
    
    if norm in [1,2]:
        dmax = 3.0  # Maximum perturbation
        lb, ub = 0., 1.  # Bounds of the attack space. Can be set to `None` for unbounded
        y_target = None  # None if `error-generic` or a class label for `error-specific`
        # Should be chosen depending on the optimization problem
        solver_params = {
            'eta': 0.5, 
            'eta_min': 2.0, 
            'eta_max': None,
            'max_iter': 100, 
            'eps': eps
        }
        
        from secml.adv.attacks import CAttackEvasionPGDLS
        pgd_ls_attack = CAttackEvasionPGDLS(classifier=clf,
                                            double_init_ds=tr,
                                            distance=noise_type, 
                                            dmax=dmax,
                                            solver_params=solver_params,
                                            y_target=y_target)
    
        
    
    print("Attack started...")
    start_time = time.time()
    eva_y_pred, _, eva_adv_ds, _ = pgd_ls_attack.run(attack_X, attack_y)
    print((time.time() - start_time)/3600,' hrs')
    print("Attack complete!")
    
    acc = metric.performance_score(
        y_true=attack_y, y_pred=clf.predict(attack_X))
    acc_attack = metric.performance_score(
        y_true=attack_y, y_pred=clf.predict(eva_adv_ds.X))
    
    print("Accuracy on test set before attack: {:.2%}".format(acc))
    print("Accuracy on test set after attack: {:.2%}".format(acc_attack))
    
    f = open(os.path.join(cwd,'PGDL'+str(norm)), 'wb') 
    pickle.dump(eva_adv_ds, f)
    
    
    return eva_adv_ds.X

def MIM(clf,attack_X,attack_y, norm,eps):
    
    ''' Perform projected-gradient method attack
    '''
    
    print(attack_X.get_data().shape)
    print(attack_y.get_data().shape)
    attack = CAttackEvasionCleverhans(clf, y_target=None, clvh_attack_class=MomentumIterativeMethod, store_var_list=None,norm=norm,eps=eps)
    
    print("Attack started...")
    start_time = time.time()
    eva_y_pred, _, eva_adv_ds, _ = attack.run(attack_X, attack_y)
    print((time.time() - start_time)/3600,' hrs')
    print("Attack complete!")
    
    f = open(os.path.join(cwd,'MIML'+str(norm)+'_'+str(eps)), 'wb')
    pickle.dump(eva_adv_ds, f)
    
    acc = metric.performance_score(
        y_true=attack_y, y_pred=clf.predict(attack_X))
    acc_attack = metric.performance_score(
        y_true=attack_y, y_pred=clf.predict(eva_adv_ds.X))
    
    print("Accuracy on test set before attack: {:.2%}".format(acc))
    print("Accuracy on test set after attack: {:.2%}".format(acc_attack))
    
    
    
    
    return eva_adv_ds.X
###############################################  EG-Booster Functions   ####################################### 

def deepexplain(clf,tr,attack_dsX,path,nb):
    
    # perform shap explanations on original images
    org_img = attack_dsX.get_data()[:nb]
    background = tr.X.get_data()[np.random.choice(tr.X.get_data().shape[0], 100, replace=False)]
    e = shap.DeepExplainer(clf.model, torch.cuda.FloatTensor(background.reshape(background.shape[0],1,28,28)))
    print('Starting Deepexplainer...')
    blockPrint()
    start_time = time.time()
    shap_values = e.shap_values(torch.cuda.FloatTensor(org_img.reshape(org_img.shape[0],1,28,28)))
    enablePrint()
    print((time.time() - start_time)/3600,' hrs')
    shp=np.array(shap_values)
    print('explanation shape: ',shp.shape)
    # save shapley values
    np.save(path, shp)
    
    
    return shp

def One_EG_Attack(org_img,adv_img,shaply,label,clf,norm, eps):
    
    '''
    org_img : original image (28,28)
    adv_img : PGD image (28,28)
    shaply: shapley values of the true label (28,28)
    label : the true label of org_img
    clf : cclassifier
    '''
    lines = 28
    cols = 28
    ege_best = adv_img.copy()
    change = 0
    pos_unperturbed = 0
    tot_perturbed = 0
    out_bound =0
    
    #print(ege_best-org_img.shape)
    nrm = round(torch.norm(torch.FloatTensor(ege_best-org_img), norm).item(),2)
    
    print('Initial perturbation size: ',nrm)
    if  nrm > eps:
        raise ValueError("norm {} cannot be used : Choose the same norm order used for initial attack. If not applicable, use a higher epsilon bound".format(norm))
    
    for i in range(lines):
        for j in range(cols):
            if adv_img[i,j] != org_img[i,j]:
                tot_perturbed+=1
            if shaply[i,j] < 0 and adv_img[i,j] != org_img[i,j]:
                # if pixel is originally negative to the original prediction but is perturbed
                ege_best[i,j] = org_img[i,j]
                if clf.predict(adv_img.reshape(1,784)).get_data()[0] != label and clf.predict(ege_best.reshape(1,784)).get_data()[0] == label:
                    ege_best[i,j] = adv_img[i,j]
                    
                else:
                    change-=1
    attmpt=0 
    if clf.predict(ege_best.reshape(1,784)).get_data()[0] == label:
                     
        for i in range(lines):
            for j in range(cols):       
                if shaply[i,j] > 0:
                    attmpt+=1
                    # if pixel is originally positive to the original prediction but not perturbed
                    if adv_img[i,j] == org_img[i,j]:
                        pos_unperturbed+=1
                        
                    if clf.predict(ege_best.reshape(1,784)).get_data()[0] == label:
                        if adv_img[i,j] != 1.0:
                            if adv_img[i,j] == org_img[i,j]:
                                change+=1
                                
                            if adv_img[i,j] <= 0.5:
                                pert = 0.4
                                ege_best[i,j]+=pert
                            if adv_img[i,j] > 0.5:
                                pert = -0.4
                                ege_best[i,j]+=pert
                                
                            attempt = 0
                            reduce = 1
                            while attempt < 20 and round(torch.norm(torch.FloatTensor(ege_best-org_img), norm).item(),2)/1000 > eps:
                                reduce +=1
                                pert = pert/reduce
                                ege_best[i,j] = adv_img[i,j] + pert
                                attempt += 2
                            
                            
                            if round(torch.norm(torch.FloatTensor(ege_best-org_img), norm).item(),2)> eps:
                                ege_best[i,j] = adv_img[i,j]
                                out_bound+=1
                    else:
                        break
        

    if attmpt!= 0:
        return ege_best, change/float(tot_perturbed) , pos_unperturbed,out_bound/attmpt
    else:
        return ege_best, change/float(tot_perturbed) , pos_unperturbed,0
    
    
def EG_Attack(eva_adv_dsX,attack_dsX,Clabels,clf,shp,attack_name,norm, eps,run,size,save=True):
    
    ''' Perform giuided attack over all adv data
    '''
    
    print('Model accuracy under unguided attack ',metric.performance_score(
        y_true=Clabels, y_pred=clf.predict(eva_adv_dsX)))
    
    out_boundAll = []
    adv_imgs = eva_adv_dsX.get_data()
    org_imgs = attack_dsX.get_data()
    best_adv = adv_imgs.copy()
    
    adv_imgs = adv_imgs.reshape(adv_imgs.shape[0],28,28)
    org_imgs = org_imgs.reshape(org_imgs.shape[0],28,28)
    
    labels = Clabels.get_data()
    print('Shape of pgd images: ',adv_imgs.shape)
    print('Shape of orginal images: ',org_imgs.shape)
    print('Model"s input shape: ',clf.input_shape)
    
    
    tot_gain = 0
    tot_pos =0
    
    start_time = time.time()
    for ind in range(org_imgs.shape[0]):
        with suppress_stdout():
            new_adv , change, pos_unpert,out_bound  = One_EG_Attack(org_imgs[ind],adv_imgs[ind],shp[labels[ind],ind,0,:,:],labels[ind],clf,norm, eps)
        
        tot_gain += change
        tot_pos += pos_unpert 
        out_boundAll.append(out_bound)
        if clf.predict(adv_imgs[ind].reshape(1,784))[0] != labels[ind].item():
            print('adv image {} is already misclassified'.format(ind))
        else:
            print('adv image {} is initially not successful'.format(ind))
        if clf.predict(new_adv.reshape(1,784))[0] != labels[ind]:
            print('Guided attack succeeded on image {} with a gain of {}'.format(ind,change))
            best_adv[ind] = new_adv.reshape(784)
        else:
            print('Guided attack did not succeed on image {} as the number of unpertubed positive features is {} and the number of out of bound perturbations is {}'.format(ind,pos_unpert,out_bound))
        if ind == size:
            print('Guided attack performed on the first {} samples'.format(size))
            break
    print((time.time() - start_time)/3600,' hrs')
    # save new adv
    if save==True:
        np.save(os.path.join(cwd,'EG-'+attack_name+'L'+str(norm)+'_run'+str(run)),best_adv)
    
    print('Model accuracy under guided attack ',metric.performance_score(
        y_true=Clabels, y_pred=clf.predict(best_adv)))
    
    
    print('Avg change :',(float(tot_gain)/org_imgs.shape[0])*100)
    print('Avg unperturbed positives :',float(tot_pos)/org_imgs.shape[0])
    print('The average rate of out of bound positive perturbations is',np.mean(out_boundAll)*100)
    
    return best_adv[:size]

def plot(image,path,label,pred,attack_name):
    
    
    # plot the sample
    if attack_name != 'clean':
        plt.title(attack_name+" {} ({})".format(label,pred),color=("green" if label != pred else "red"))
    else:
        plt.title(attack_name+" {} ({})".format(label,pred))
        
    plt.imshow(image.reshape((28, 28)), cmap='gray')
    plt.savefig(path)
    
def similarity(run1,run2):
    
    if run1.shape[0] != run2.shape[0]:
        raise ValueError("run1 shape {} is different than run2 shape {}".format(run1.shape,run2.shape))
    
    for i in range(run1.shape[0]):
        if run1[i] == run2[i]:
            i+=1
    
    sim = (run1*run2>=0).sum()/np.prod(list(run1.shape))   
    
    return sim

def similarity_shp(shp1,shp2,l,x_len,y):
    
    '''Compute similarities of SHAP explanations'''
    sim = []
    for ind in range(x_len): # across different samples
        
        shp1_c=shp1[y[ind],ind].reshape((np.prod(list(shp1[y[ind],ind].shape))))
        shp2_c=shp2[y[ind],ind].reshape((np.prod(list(shp2[y[ind],ind].shape))))
        
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

def stability(k,ts,l,attack,norm,eps,clf,size,runs):
    '''Compute the stability of EG-Booster'''
    
    if runs == []:
        # Load results of baseline attack
        f = open(os.path.join(cwd,attack), 'rb')
        eva_adv_ds = pickle.load(f)
        eva_adv_ds = eva_adv_ds[:size, :]
        ts = ts[:size, :]
        # run EG-Booster k times
        y=ts.Y.get_data()[:size]
        
        
        for i in range(k):
            print('Run {} of EG-bosster: '.format(i))
            shp=np.load(os.path.join(cwd,'org_exp'+str(i)+'.npy'))
            ege_best=EG_Attack(eva_adv_ds.X,ts.X,ts.Y,clf,shp,attack,norm,eps,run=1,size=size)
            runs.append(clf.predict(ege_best[:size]))
    elif len(runs) != k:
        raise ValueError('Given list of runs length ({}) should be equal to k ({})'.format(len(runs),k))
    y=ts.Y.get_data()[:size]
    stb_shp=[]
    stb_eg=[]
    for i in range(k):
        s_eg=[]
        s_shp=[]
        for j in range(k):
            if i!=j:
                print('Computing similarities of run {} with run {}'.format(i,j))
                base_run=np.load(os.path.join(cwd,'org_exp'+str(i)+'.npy'))
                temp_run=np.load(os.path.join(cwd,'org_exp'+str(j)+'.npy')) 
                s_eg.append(similarity(runs[i],runs[j]))
                
                s_shp.append(similarity_shp(base_run,temp_run,l,len(y),y))
               
        print('Current similarity of shap: ',np.mean(s_shp))
        print('Current similarity of shap: ',np.mean(s_eg))
        stb_shp.append(np.mean(s_shp))
        stb_eg.append(np.mean(s_eg))
                
    return np.mean(stb_eg), np.mean(stb_shp)
    
def stability_plot(k,ts,l,attack,norm,eps,clf,size,path):
    
    runs = []
    # Load results of baseline attack
    f = open(os.path.join(cwd,attack+'L'+str(norm)), 'rb')
    eva_adv_ds = pickle.load(f)
    eva_adv_ds = eva_adv_ds[:size, :]
    ts = ts[:size, :]
    # run EG-Booster k times
    for i in range(k):
        print('Run {} of EG-bosster: '.format(i))
        shp=np.load(os.path.join(cwd,'org_exp'+str(i)+'.npy'))
        ege_best=EG_Attack(eva_adv_ds.X,ts.X,ts.Y,clf,shp,attack,norm,eps,run=1,size=size)
        runs.append(clf.predict(ege_best[:size]))
        
    shp_stab=[]
    eg_stab=[]
    
    for k_i in range(k):
        if k_i in [0,1]:
            shp_stab.append(1.0)
            eg_stab.append(1.0)
        else:
            stb_eg_k, stb_shp_k = stability(k_i,ts,l,attack,norm,eps,clf,size,runs[:k_i])
            shp_stab.append(stb_shp_k)
            eg_stab.append(stb_eg_k)
     
    x=range(k)
    plt.plot(x, shp_stab,color='red',label='SHAP')
    plt.plot(x, eg_stab,color='blue',label='EG-Booster')
    plt.legend()
    plt.xlabel("k")
    plt.ylabel("stability")
    plt.savefig(path)
    
def eps_plot(ts,size,clf,path,attack):
    
    '''Perform EG-Booster analysis for diferent epsilons'''
    
    attack_ds = ts[:size, :]
    shp=np.load(os.path.join(cwd,'org_exp'+str(0)+'.npy'))
    epsilons = [0.1,0.3,0.5,0.6,0.7] # if MIM exclude 0.5
    if attack == 'MIM':
        epsilons = [0.1,0.3,0.6,0.7] 
    base_rate=[]
    booster_rate=[]
    for eps in epsilons:
        print('eps=',eps)
        
        if attack == 'MIM':
            adv_imgs = MIM(clf,attack_ds.X,attack_ds.Y, np.inf,eps)
        
        elif attack == 'FGS':
                adv_imgs = FGS(clf,attack_ds.X,attack_ds.Y,np.inf,eps)
                
        elif attack == 'PGD':
                adv_imgs = pgd(clf,attack_ds.X,attack_ds.Y,ts,np.inf,eps)
                
                
        elif attack == 'BIM':
            adv_imgs = BIM(clf,attack_ds.X,attack_ds.Y,np.inf,eps)
            
        eva_adv_ds = CDataset(adv_imgs,attack_ds.Y)
        base_rate.append(1-metric.performance_score(y_true=attack_ds.Y, y_pred=clf.predict(eva_adv_ds.X.get_data()[:size])))
        if attack=='MIM':
            ege_best=np.load(os.path.join(cwd,'EG-MIM_'+str(eps)+'Linf_run2.npy'))
        else:
            ege_best=EG_Attack(eva_adv_ds.X,attack_ds.X,attack_ds.Y,clf,shp,attack,np.inf,eps,run=2,size=size,save=True)
        booster_rate.append(1-metric.performance_score(y_true=attack_ds.Y, y_pred=clf.predict(ege_best)))
    
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
    
    

def main(_):
    
    
    print('Loading data')
    loader = CDataLoaderMNIST()
    tr = loader.load('training')
    ts = loader.load('testing')
    
    
    # Normilize data
    tr.X /= 255.0
    ts.X /= 255.0
    
    print('Loading and testing the CNN Model')
    with suppress_stdout():
        clf = load_model('mnist-cnn')
        label_torch = clf.predict(ts.X, return_decision_function=False)
        acc_torch = metric.performance_score(ts.Y, label_torch)
    print("Model Accuracy: {}".format(acc_torch))

    ### Perfom FGSLinf Attack
    attack_ds = ts[:FLAGS.size, :]
    print('\nPerforming baseline attack')
    adv_imgs = FGS(clf,attack_ds.X,attack_ds.Y,FLAGS.norm,FLAGS.eps)
    with suppress_stdout():
        eva_adv_ds = CDataset(adv_imgs,attack_ds.Y)
    
    ### Perform explanations for EG-Booster
    shp = deepexplain(clf,tr,attack_ds.X,os.path.join(cwd,'org_exp0'),FLAGS.size)
    
    ### perfom EG-Booster
    print('\nPerforming EG-Booster')
    ege_best = EG_Attack(eva_adv_ds.X,attack_ds.X,attack_ds.Y,clf,shp,'FGS',norm=FLAGS.norm,eps=FLAGS.eps,run=2,size=FLAGS.size)
    
    #### Stability plots
    size=1000
    ### perform shap explanations 10 times for stability analysis
    print('\nStability Analysis')
    for i in range(10):
        shp = deepexplain(clf,tr,attack_ds.X,os.path.join(cwd,'org_exp'+str(i)),size)
        
    stability_plot(10,ts,10,'FGS',FLAGS.norm,FLAGS.eps,clf,size,os.path.join(cwd,'stability'+'_FGSLinf.png'))
    
    #### Evasion curve plots
    print('\nPloting Evasion curve for different epsilons')
    size=5000
    eps_plot(ts,size,clf,os.path.join(cwd,'eps_anaFGS.png'),'FGS')
    
    
if __name__ == '__main__':    
    flags.DEFINE_integer("size", 10000, "Test set size")
    flags.DEFINE_float("eps", 0.3, "Total epsilon for FGS and PGD attacks.")
    flags.DEFINE_float("norm",np.inf, "Used distance metric.")
    flags.DEFINE_bool(
        "defended", False, "Use the undefended model"
    )

    app.run(main)
