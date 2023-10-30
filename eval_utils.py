import os
from sklearn import metrics
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
import train_utils
Abnormal_type=[     'Abuse','Arrest','Arson','Assault','Burglary',
                    'Explosion','Fighting','Normal','RoadAccidents','Robbery',
                    'Shooting','Shoplifting','Stealing','Vandalism' ]
np.set_printoptions(threshold=np.inf)
def cal_f1(scores,labels):
    return metrics.f1_score(labels,scores)

def cal_rmse(scores,labels):
    return metrics.mean_squared_error(labels,scores)**0.5

def cal_confusion_matrix(predict,labels,tag):
    # predict, labels is [B,class_num]
    predict=torch.argmax(predict,dim=1,keepdim=False).cpu().numpy()
    labels=torch.argmax(labels,dim=1,keepdim=False).cpu().numpy()
    return metrics.confusion_matrix(labels,predict,labels=tag)

def cal_pr_auc(scores,labels):
    precision, recall, thresholds = metrics.precision_recall_curve(labels, scores,pos_label=1)
    auc = metrics.auc(recall, precision)
    plt.plot(recall, precision, 'b', label='pr = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Receiver operating characteristic example')
    plt.savefig('pr.png')
    plt.show()
    plt.close()

    return auc

def cal_auc(scores,labels):
    fpr, tpr, thresholds=metrics.roc_curve(labels,scores,pos_label=1)
    auc=metrics.auc(fpr,tpr)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')  # 横坐标是fpr
    plt.ylabel('True Positive Rate')  # 纵坐标是tpr
    plt.title('Receiver operating characteristic example')
    plt.savefig('roc.png')
    plt.show()
    plt.close()
    return auc

def cal_false_alarm(scores,labels,threshold=0.5):   # 把正常的分类为异常的比例
    ori_scores = scores
    scores=np.array([1 if score>threshold else 0 for score in scores],dtype=float)

    # false_num=0.
    # _len=len(labels)
    # for score,label in zip(scores,labels):
    #     if label!=score:
    #         false_num+=1
    fp=np.sum(scores*(1-labels))
    # train_utils.log(ori_scores[np.where(scores*(1-labels)>0)])
    # print(ori_scores[np.where(scores*(1-labels)>0)])
    # keys = np.load('SHT_Test_keys.npy', allow_pickle=True)
    # keys = np.repeat(keys, 16)
    # error_keys = keys[np.where(scores*(1-labels)>0)]
    # # print(error_keys)
    # # error_keys = [key for i, key in enumerate(error_keys) if i % 16 == 0]
    # txt_path = 'errorkeys_2.txt'
    # if os.path.exists(txt_path):
    #     os.remove(txt_path)
    # np.savetxt(txt_path, error_keys, fmt='%s')
    # print(np.sum(1-labels))
    return fp/np.sum(1-labels)

def cal_false_neg(scores,labels,threshold=0.5):
    ori_scores = scores
    scores=np.array([1 if score>threshold else 0 for score in scores],dtype=float)
    fn=np.sum((1-scores)*(labels))
    # print(ori_scores[np.where((1-scores)*(labels) > 0)])
    #
    # # train_utils.log(ori_scores[np.where((1-scores)*(labels) > 0)])
    # keys = np.load('SHT_Test_keys.npy', allow_pickle=True)
    # keys = np.repeat(keys, 16)
    # error_keys_neg = keys[np.where((1-scores)*(labels) > 0)]
    # # print(error_keys_neg)
    # # error_keys_neg = [key for i, key in enumerate(error_keys_neg) if i % 16 == 0]
    # txt_path = 'errorkeys_neg2.txt'
    # if os.path.exists(txt_path):
    #     os.remove(txt_path)
    # np.savetxt(txt_path, error_keys_neg, fmt='%s')
    # print(np.sum(labels))
    return fn/np.sum(labels)

def cal_precision(scores,labels,threshold=0.5):
    scores=np.array([1 if score>threshold else 0 for score in scores],dtype=float)
    tp=np.sum(scores*labels)
    return tp/np.sum(scores)

def cal_accuracy(scores,labels,threshold=0.5):
    scores=np.array([1 if score>threshold else 0 for score in scores],dtype=float)
    tp=np.sum(scores*labels)
    tn=np.sum((1-scores)*(1-labels))
    return np.sum(tp+tn)/scores.shape[0]


def cal_recall(scores,labels,threshold=0.5):
    scores=np.array([1 if score>threshold else 0 for score in scores],dtype=float)
    tp=np.sum(scores*labels)
    fn=np.sum((1-scores)*labels)

    return tp/(tp+fn)

def cal_specific(scores,labels,threshold=0.5):
    scores=np.array([1 if score>threshold else 0 for score in scores],dtype=float)
    tn=np.sum((1-labels)*(1-scores))
    return tn/np.sum(1-labels)

def cal_sensitivity(scores,labels,threshold=0.50):
    scores=np.array([1 if score>threshold else 0 for score in scores],dtype=float)
    tp=np.sum(scores*labels)
    return tp/np.sum(labels)

def cal_score_gap(scores,labels):
    labels=labels.astype(bool)
    neg_labels=(1-labels).astype(bool)
    positive=np.mean(scores[labels])
    negative=np.mean(scores[neg_labels])
    return positive-negative

def cal_geometric_mean(scores,labels,threshold=0.5):
    tn=cal_specific(scores,labels,threshold)
    tp=cal_sensitivity(scores,labels,threshold)
    return np.sqrt(tp*tn)

def cal_f_measure(scores,labels,threshold=0.5):
    p=cal_precision(scores,labels,threshold)
    r=cal_recall(scores,labels,threshold)
    return 2*p*r/(p+r)

def cal_MCC(scores,labels,threshold=0.5):
    scores=np.array([1 if score>threshold else 0 for score in scores],dtype=float)
    tp=np.sum(scores*labels)
    tn=np.sum((1-scores)*(1-labels))
    fp=np.sum(scores*(1-labels))
    fn=np.sum((1-scores)*labels)
    return (tp*tn-fp*fn)/np.sqrt((tp+fp)*(fp+fn)*(tn+fp)*(tn+fn))

def cal_pAUC(scores,labels):
    sum_gt_p=np.sum(labels)
    sum_gt_n=labels.shape[0]-sum_gt_p
    sum_pred_p=np.sum(scores[labels.astype(bool)])
    sum_pred_n=np.sum(scores[(1-labels).astype(bool)])
    return 0.5*(sum_pred_p/sum_gt_p-sum_pred_n/sum_gt_n+1)

def eval_each_part(labels_dict,scores_dict,logger=None):
    map = 0
    for key in labels_dict.keys():
        score=scores_dict[key]
        if key=='Normal':
            auc='None'
            pr_auc='None'
            gap='None'
            false_alarm=cal_false_alarm(np.array(score,dtype=float),np.array(labels_dict[key],dtype=float))
            normal_far=false_alarm
            if logger == None:
                print('{}: \tAUC \t{}, PR-AUC \t{}, FAR \t{}\tGAP\t{}'.format(key, auc, pr_auc, normal_far, gap))
            else:
                logger.info('{}: \tAUC \t{}, PR-AUC \t{}, FAR \t{}\tGAP\t{}'.format(key,auc,pr_auc,normal_far,gap))
        else:
            # print(type,np.array(score).shape,np.array((labels_dict[type])).shape)
            auc=cal_auc(np.array(score,dtype=float),np.array(labels_dict[key],dtype=float))
            pr_auc=cal_pr_auc(np.array(score,dtype=float),np.array(labels_dict[key],dtype=float))
            map+=pr_auc
            gap=cal_score_gap(np.array(score,dtype=float),np.array(labels_dict[key],dtype=float))
            false_alarm=cal_false_alarm(np.array(score,dtype=float),np.array(labels_dict[key],dtype=float))
            if logger==None:
                print('{}: \tAUC \t{:.4f}, PR-AUC \t{:.4f}, FAR \t{}\tGAP\t{:.4f}'.format(key,auc,pr_auc,false_alarm,gap))
            else:
                logger.info('{}: \tAUC \t{:.4f}, PR-AUC \t{:.4f}, FAR \t{}\tGAP\t{:.4f}'.format(key,auc,pr_auc,false_alarm,gap))
    return normal_far,map/13

def eval_classification(logits,labels):
    # the labels here is the int label
    # labels=torch.argmax(labels,dim=1).to(logits.device)
    logits=(torch.argmax(logits,dim=1))


    a=torch.le(labels,logits).float()
    b=torch.lt(labels,logits).float()
    accuracy_top_1=torch.mean(a-b)
    #maxk=max((5,))
    #_,pred=logits.topk(maxk,dim=1,largest=True)

    return accuracy_top_1

def eval_classification_binary(logits,labels):
    # all of them are with [N]
    N=logits.shape[0]
    positive_true=torch.nonzero(labels*F.relu(logits-0.5)).shape[0]
    negative_true=torch.nonzero(((1-labels)*F.relu(0.5-logits))).shape[0]
    return (positive_true+negative_true)/N

def eval(total_scores,total_labels):
    total_scores = np.array(total_scores)
    total_labels = np.array(total_labels)

    # logger.info('===================')
    auc = cal_auc(total_scores, total_labels)
    pr_auc = cal_pr_auc(total_scores, total_labels)
    far = cal_false_alarm(total_scores, total_labels)
    fanr = cal_false_neg(total_scores, total_labels)
    gap = cal_score_gap(total_scores, total_labels)
    train_utils.log('AUC {:.2f}%, PR_AUC {:.2f}, FAR {:.2f}%, FANR {:.2f}%, GAP {:.2f}%'.format(auc * 100, pr_auc *100, far * 100, fanr*100, gap * 100))

    return auc
