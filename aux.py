### Auxiliary functions for learner to use
def AUC(soft_predList,labelList):
    """
    Use the probabilities
    """
    # pdb.set_trace()
    if isinstance(soft_predList,list):
        soft_predList = np.concatenate(soft_predList,0)
    if isinstance(labelList,list):
        labelList = np.concatenate(labelList,0)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(labelList, soft_predList[:,1], pos_label = 1)
    auc_roc = sklearn.metrics.auc(fpr, tpr)
    precision, recall, threshold = sklearn.metrics.precision_recall_curve(labelList, soft_predList[:,1], pos_label = 1)
    auc_prc = sklearn.metrics.auc(recall,precision)
    #save fpr & tpr for plotting
    np.savetxt('FprTpr_'+saveName.split('.')[0]+ '.csv',np.array([fpr,tpr]), delimiter = ',')
    return auc_roc, auc_prc

def globalAcc(predList,labelList):
    if not isinstance(predList,torch.Tensor):
        predList = torch.cat(predList)
    if not isinstance(labelList,torch.Tensor):
        labelList = torch.cat(labelList)
    #pdb.set_trace()
    acc = torch.sum(predList==labelList).float()/( predList.shape[0] )
    return acc

