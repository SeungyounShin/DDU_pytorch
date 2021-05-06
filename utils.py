from sklearn.metrics import precision_recall_curve,roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def test_result(test_res,true_len):
    true = np.array(test_res['y_true'])
    acc = test_res['n_correct']/true_len
    precision2, recall2, _   = precision_recall_curve(test_res['y_true'],test_res['epis_'])
    precision2=np.flip(precision2)
    recall2=np.flip(recall2)
    #print(recall[-10:],precision[-10:])
    aupr2 = np.trapz(np.asarray(precision2),np.asarray(recall2))
    auroc2 = roc_auc_score(test_res['y_true'],np.array(test_res['epis_']))

    precision, recall, _   = precision_recall_curve(test_res['y_true'],test_res['alea_'])
    precision=np.flip(precision)
    recall=np.flip(recall)
    #print(recall[-10:],precision[-10:])
    aupr3 = np.trapz(np.asarray(precision),np.asarray(recall))
    auroc3 = roc_auc_score(test_res['y_true'],np.array(test_res['alea_']))

    temp=np.array(test_res['y_probs'])
    temp2 = np.array(test_res['y_true'])

    temp3 = temp[np.where(temp2==0)]
    temp4 = temp[np.where(temp2==1)]
    precision, recall, th   = precision_recall_curve(temp2,temp,pos_label=1)
    precision=np.flip(precision)
    recall=np.flip(recall)
    aupr1 = np.trapz(np.asarray(precision),np.asarray(recall))
    auroc1 = roc_auc_score(np.array(test_res['y_true']),np.array(test_res['y_probs']))
    return acc,auroc1,auroc2,auroc3,aupr1,aupr2,aupr3 #max softmax epis alea

def plot_res(test_res,true_len,loss_name,idx,rs,rp,train_log):
    plt.cla()
    plt.clf()

    [lacc,lauroc1,laupr1,lauroc2,laupr2]=train_log
    true = np.array(test_res['y_true'])
    acc = test_res['n_correct']/true_len
    #print(acc)
    precision1, recall1, _   = precision_recall_curve(test_res['y_true'],np.array(test_res['alea_']))
    precision1=np.flip(precision1)
    recall1=np.flip(recall1)
    #print(true[-10:],np.array(test_res['alea_'])[-10:])
    aupr1 = np.trapz(np.asarray(precision1),np.asarray(recall1))
    auroc1 = roc_auc_score(test_res['y_true'],np.array(test_res['alea_']))
    plt.plot(recall1, precision1, lw=2, c='b', label='[alea] aupr:{%.3f} auroc: {%.3f}'%(float(aupr1),float(auroc1)))

    precision2, recall2, _   = precision_recall_curve(test_res['y_true'],test_res['epis_'])
    precision2=np.flip(precision2)
    recall2=np.flip(recall2)
    #print(recall[-10:],precision[-10:])
    aupr2 = np.trapz(np.asarray(precision2),np.asarray(recall2))
    auroc2 = roc_auc_score(test_res['y_true'],np.array(test_res['epis_']))

    plt.figure(1)
    plt.clf()
    plt.plot(recall2, precision2, lw=2, c='r', label='[epis] aupr:{%.3f} auroc: {%.3f}'%(float(aupr2),float(auroc2)))
    plt.title('recall and precision -> acc:'+str(acc)+ 'rs: '+str(rs)+' ,rp: '+str(rp))
    plt.legend()
    plt.savefig('./figure/0505/mln_cifar_OOD_PR cruve_'+str(idx)+'.png')

    plt.figure(2)
    plt.clf()
    temp=np.array(test_res['alea_'])
    temp3 = temp[np.where(true==0)]
    temp4 = temp[np.where(true==1)]
    plt.hist(temp3, color='b',label='in distribution',bins=100, alpha=0.5)
    plt.hist(temp4,color='r',label='out distribution',bins=100, alpha=0.5)
    plt.title(' alea density plot -> rs: '+str(rs)+' ,rp: '+str(rp))
    plt.legend()
    plt.savefig('./figure/0505/mln_cifar_OOD_alea_hist_'+str(idx)+'.png')

    plt.figure(3)
    plt.clf()
    #print(" plot epis len :: ",len(test_res['epis_']))
    temp=np.array(test_res['epis_'])
    temp3 = temp[np.where(true==0)]
    temp4 = temp[np.where(true==1)]
    #print(temp4.shape,temp3.shape)
    plt.hist(temp3, color='b',label='in distribution',bins=100, alpha=0.5)
    plt.hist(temp4,color='r',label='out distribution',bins=100, alpha=0.5)
    plt.title(' epis density plot -> : rs'+str(rs)+' ,rp: '+str(rp))
    plt.legend()
    plt.savefig('./figure/0505/mln_cifar_OOD_epis_hist_'+str(idx)+'.png')


    temp=np.array(test_res['y_probs'])
    temp2 = np.array(test_res['y_true'])

    temp3 = temp[np.where(temp2==0)]
    temp4 = temp[np.where(temp2==1)]
    precision, recall, th   = precision_recall_curve(temp2,temp,pos_label=1)
    precision=np.flip(precision)
    recall=np.flip(recall)
    aupr = np.trapz(np.asarray(precision),np.asarray(recall))
    auroc = roc_auc_score(np.array(test_res['y_true']),np.array(test_res['y_probs']))
    #print('AUPR: {}, AUROC: {}'.format(aupr,auroc))
    plt.figure(4)
    plt.clf()
    plt.plot(recall, precision, lw=2, c='b', label= 'aupr:{%.3f} auroc: {%.3f}'%(float(aupr),float(auroc)))
    plt.title('max softmax recall and precision -> rs:'+str(rs)+' ,rp: '+str(rp))
    plt.legend()
    plt.savefig('./figure/0505/maxsoftmax_cifar_OOD_PR_Curve_'+str(idx)+'.png')

    plt.figure(5)
    plt.clf()
    plt.hist(temp3, color='b',label='in distribution',bins=100, alpha=0.5)
    plt.hist(temp4,color='r',label='out distribution',bins=100, alpha=0.5)
    plt.title('max softmax density plot -> rs: '+str(rs)+' ,rp: '+str(rp))
    plt.legend()
    plt.savefig('./figure/0505/maxsoftmax_cifar_OOD_hist_'+str(idx)+'.png')

    plt.figure(6)
    plt.clf()
    fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(5, 1, constrained_layout=True, sharey=True)

    #ax = plt.subplot(511)
    ax1.set_title("accuracy")
    ax1.plot(lacc)

    #ax = plt.subplot(512)
    ax2.set_title("max softmax auroc")
    ax2.plot(lauroc1)

    #ax = plt.subplot(513)
    ax3.set_title("max softmax aupr")
    ax3.plot(laupr1)

    #ax = plt.subplot(514)
    ax4.set_title("epis auroc")
    ax4.plot(lauroc2)

    #ax = plt.subplot(515)
    ax5.set_title("epis aupr")
    ax5.plot(laupr2)

    fig.suptitle("training")
    plt.savefig('./figure/0505/train_log_cifar_OOD_'+str(idx)+'.png')

def print_n_txt(_f,_chars,_addNewLine=True,_DO_PRINT=True):
    if _addNewLine: _f.write(_chars+'\n')
    else: _f.write(_chars)
    _f.flush();os.fsync(_f.fileno()) # Write to txt
    if _DO_PRINT:
        print (_chars)
