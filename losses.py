#!/usr/bin/python
#****************************************************************#
# ScriptName: losses.py
# Author: @alibaba-inc.com
# Create Date: 2024-03-05 16:25
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2024-06-26 20:56
# Function: 
#***************************************************************#

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from utils import print_rank_0

class OrdinalRegressionLoss(nn.Module):

    def __init__(self, num_class, train_cutpoints=False, scale=18.0):
        super().__init__()
        self.num_classes = num_class
        num_cutpoints = self.num_classes - 1
        self.cutpoints = torch.arange(num_cutpoints).float()*scale/(num_class-2) - scale / 2
        #self.cutpoints = nn.Parameter(self.cutpoints)
        #if not train_cutpoints:
        #    self.cutpoints.requires_grad_(False)

    def forward(self, pred, semantic_labels_softmax):
        sigmoids = torch.sigmoid((self.cutpoints.to(pred.device) - pred) * 1)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat((
                sigmoids[:, [0]],
                link_mat,
                (1 - sigmoids[:, [-1]])
            ),
            dim=1
        )

        eps = 1e-15
        likelihoods = torch.clamp(link_mat, eps, 1 - eps)

        neg_log_likelihood = torch.log(likelihoods)

        #loss = -torch.gather(neg_log_likelihood, 1, label.to(pred.device)).mean()
        loss = -torch.sum(neg_log_likelihood * semantic_labels_softmax, 1).mean()
            
        return loss, likelihoods

def coral_loss(logits, levels, imp):
    val = (-torch.sum((F.logsigmoid(logits)*levels
                      + (F.logsigmoid(logits) - logits)*(1-levels)) * imp, dim=1))
    return torch.mean(val)

def coral_loss_v2(logits, levels, imp):
    #th = torch.tensor([-1.386, -0.405, 0.405, 1.386]).to(logits.device)
    th = torch.tensor([-0.3, -0.1, 0.1, 0.3]).to(logits.device)
    re_center_logits = F.sigmoid(logits) - th
    eps = 1e-15
    re_center_logits = torch.clamp(re_center_logits, eps, 1 - eps)

    val = (-torch.sum((torch.log(re_center_logits)*levels
                      + torch.log(1 - re_center_logits)*(1-levels)) * imp, dim=1))
    return torch.mean(val)

def corn_loss(logits, y_train, num_classes):
    """Computes the CORN loss described in our forthcoming
    'Deep Neural Networks for Rank Consistent Ordinal
    Regression based on Conditional Probabilities'
    manuscript.

    Parameters
    ----------
    logits : torch.tensor, shape=(num_examples, num_classes-1)
        Outputs of the CORN layer.

    y_train : torch.tensor, shape=(num_examples)
        Torch tensor containing the class labels.

    num_classes : int
        Number of unique class labels (class labels should start at 0).

    Returns
    ----------
        loss : torch.tensor
        A torch.tensor containing a single loss value.

    Examples
    ----------
    >>> import torch
    >>> from coral_pytorch.losses import corn_loss
    >>> # Consider 8 training examples
    >>> _  = torch.manual_seed(123)
    >>> X_train = torch.rand(8, 99)
    >>> y_train = torch.tensor([0, 1, 2, 2, 2, 3, 4, 4])
    >>> NUM_CLASSES = 5
    >>> #
    >>> #
    >>> # def __init__(self):
    >>> corn_net = torch.nn.Linear(99, NUM_CLASSES-1)
    >>> #
    >>> #
    >>> # def forward(self, X_train):
    >>> logits = corn_net(X_train)
    >>> logits.shape
    torch.Size([8, 4])
    >>> corn_loss(logits, y_train, NUM_CLASSES)
    tensor(0.7127, grad_fn=<DivBackward0>)
    """
    sets = []
    for i in range(num_classes-1):
        label_mask = y_train > i-1
        label_tensor = (y_train[label_mask] > i).to(torch.int64)
        sets.append((label_mask, label_tensor))

    num_examples = 0
    losses = 0.
    for task_index, s in enumerate(sets):
        train_examples = s[0]
        train_labels = s[1]

        if len(train_labels) < 1:
            continue

        num_examples += len(train_labels)
        pred = logits[train_examples, task_index]

        loss = -torch.sum(F.logsigmoid(pred)*train_labels
                          + (F.logsigmoid(pred) - pred)*(1-train_labels))
        losses += loss

    return losses/num_examples

def OR_loss(logits, labels):
    logits = logits.view(-1, 4, 2)
    val = (-torch.sum((F.log_softmax(logits, dim=2)[:, :, 1]*labels
                      + F.log_softmax(logits, dim=2)[:, :, 0]*(1-labels)), dim=1))
    return torch.mean(val)


def CondorOrdinalCrossEntropy(logits, levels, importance_weights=None, reduction='mean'):
    """computes the condor loss described in

    condor tbd.

    parameters
    ----------
    logits : torch.tensor, shape(num_examples, num_classes-1)
        outputs of the condor layer.

    levels : torch.tensor, shape(num_examples, num_classes-1)
        true labels represented as extended binary vectors
        (via `condor_pytorch.dataset.levels_from_labelbatch`).

    importance_weights : torch.tensor, shape=(num_classes-1,) (default=none)
        optional weights for the different labels in levels.
        a tensor of ones, i.e.,
        `torch.ones(num_classes-1, dtype=torch.float32)`
        will result in uniform weights that have the same effect as none.

    reduction : str or none (default='mean')
        if 'mean' or 'sum', returns the averaged or summed loss value across
        all data points (rows) in logits. if none, returns a vector of
        shape (num_examples,)

    returns
    ----------
        loss : torch.tensor
        a torch.tensor containing a single loss value (if `reduction='mean'` or '`sum'`)
        or a loss value for each data record (if `reduction=none`).

    examples
    ----------
    >>> import torch
    >>> levels = torch.tensor(
    ...    [[1., 1., 0., 0.],
    ...     [1., 0., 0., 0.],
    ...    [1., 1., 1., 1.]])
    >>> logits = torch.tensor(
    ...    [[2.1, 1.8, -2.1, -1.8],
    ...     [1.9, -1., -1.5, -1.3],
    ...     [1.9, 1.8, 1.7, 1.6]])
    >>> CondorOrdinalCrossEntropy(logits, levels)
    tensor(0.8259)
    """

    if not logits.shape == levels.shape:
        raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
                         % (logits.shape, levels.shape))

    logprobs = torch.cumsum(F.logsigmoid(logits), dim = 1)
    term1 = (logprobs*levels
             + torch.log(1 - torch.exp(logprobs)+torch.finfo(torch.float32).eps)*(1-levels))

    if importance_weights is not None:
        term1 *= importance_weights

    val = (-torch.sum(term1, dim=1))

    if reduction == 'mean':
        loss = torch.mean(val)
    elif reduction == 'sum':
        loss = torch.sum(val)
    elif reduction is None:
        loss = val
    else:
        s = ('Invalid value for `reduction`. Should be "mean", '
             '"sum", or None. Got %s' % reduction)
        raise ValueError(s)

    return loss

class AM_Softmax_v2(nn.Module): #requires classification layer for normalization 
    def __init__(self, m=0.2, s=30, d=2048, num_classes=5, use_gpu=True , epsilon=0.1):
        super(AM_Softmax_v2, self).__init__()
        self.m = m
        self.s = s 
        self.num_classes = num_classes
        self.CrossEntropy = CrossEntropyLoss()  # target-label onehot or int-tensor / logits not need softmax!

    def forward(self, features, multi_label_softmax , classifier):
        '''
        x : feature vector : (b x  d) b= batch size d = dimension 
        labels : (b,)
        classifier : Fully Connected weights of classification layer (dxC), C is the number of classes: represents the vectors for class
        '''
        # x = torch.rand(32,2048)
        # label = torch.tensor([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,])
        labels = torch.argmax(multi_label_softmax, -1)
        features = nn.functional.normalize(features, p=2, dim=1) # normalize the features
        with torch.no_grad():
            classifier.weight.div_(torch.norm(classifier.weight, dim=1, keepdim=True))

        cos_angle = classifier(features)
        cos_angle = torch.clamp( cos_angle , min = -1 , max = 1 ) 
        b = features.size(0)
        for i in range(b):
            cos_angle[i][labels[i]] = cos_angle[i][labels[i]]  - self.m 
        weighted_cos_angle = self.s * cos_angle
        log_probs = self.CrossEntropy(weighted_cos_angle , multi_label_softmax)
        return log_probs

def AM_Softmax(features, multi_label_softmax , classifier):
    '''
    x : feature vector : (b x  d) b= batch size d = dimension 
    labels : (b,)
    classifier : Fully Connected weights of classification layer (dxC), C is the number of classes: represents the vectors for class
    '''
    m = 0.2
    s = 30
    num_classes = 5
    CrossEntropy = CrossEntropyLoss()
    # x = torch.rand(32,2048)
    # label = torch.tensor([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,])
    labels = torch.argmax(multi_label_softmax, -1)
    features = nn.functional.normalize(features, p=2, dim=1) # normalize the features
    with torch.no_grad():
        classifier.weight.div_(torch.norm(classifier.weight, dim=1, keepdim=True))

    cos_angle = classifier(features)
    cos_angle = torch.clamp( cos_angle , min = -1 , max = 1 ) 
    b = features.size(0)
    for i in range(b):
        cos_angle[i][labels[i]] = cos_angle[i][labels[i]]  - m 
    weighted_cos_angle = s * cos_angle
    log_probs = CrossEntropy(weighted_cos_angle , multi_label_softmax)
    return log_probs

class AMSoftmax(nn.Module):
    def __init__(self,
                 in_feats=1024,
                 n_classes=5,
                 m=0.2,
                 s=30):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, multi_label_softmax):
        lb = torch.argmax(multi_label_softmax, -1)
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        
        costh = torch.mm(x_norm, w_norm.to(x_norm.device))
        lb_view = lb.view(-1, 1)
        if lb_view.is_cuda: lb_view = lb_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, multi_label_softmax)
        return loss, costh

class RMLoss(nn.Module):
    """ """

    def __init__(
        self,
        reduction="mean",
        beta=0.001,
    ):
        super().__init__()
        self.reduction = reduction
        self.beta = beta
        self.criterion = nn.MarginRankingLoss(margin=0.85, reduce=False)

    def forward(
        self,
        logits,
        labels,
        k_lens=None,
    ):
        total_loss = []
        indices = list(zip(k_lens[:-1], k_lens[1:]))
        #print_rank_0("---------------")
        #print_rank_0(k_lens)
        #print_rank_0(labels)
        #print_rank_0("labels_shape: {}".format(labels.shape))
        #print_rank_0("logits_shape: {}".format(logits.shape))
        for start, end in indices:
            #print_rank_0(start)
            #print_rank_0(end)
            combinations = torch.combinations(
                torch.arange(start, end, device=logits.device), 2
            )
            positive = logits[combinations[:, 0]]
            negative = logits[combinations[:, 1]]

            pos_label = labels[combinations[:, 0]]
            neg_label = labels[combinations[:, 1]]

            l2_loss = self.beta * 0.5 * (positive**2 + negative**2)
            #l2_loss = 0.7 * 0.5 * (torch.maximum((positive - (pos_label / 5 + 0.1))**2 - 0.01,  torch.tensor(0)) + torch.maximum((negative - (neg_label / 5 + 0.1))**2 - 0.01,  torch.tensor(0))) # point loss
            #print_rank_0("point_loss_shape: {}".format(l2_loss.shape))
            #print_rank_0(l2_loss)

            pw_label = torch.greater(pos_label, neg_label).to(torch.float32) * 2 - 1
            #loss = -1 * nn.functional.logsigmoid(pw_label * (positive - negative))
            loss = self.criterion(positive, negative, torch.reshape(pw_label, [-1, 1]))
            #print_rank_0("positive_shape: {}".format(positive.shape))
            #print_rank_0("pw_label: {}".format(pw_label.shape))
            #print_rank_0("loss_shape: {}".format(loss.shape))
            zero_loss = torch.zeros_like(loss)
            loss = torch.where(torch.eq(pos_label, neg_label), zero_loss, loss)
            #print_rank_0(loss)
            loss_valid = loss[torch.ne(pos_label, neg_label)]
            l2_loss_valid = l2_loss[torch.ne(pos_label, neg_label)]

            #print_rank_0("loss: {}".format(loss.mean()))
            #print_rank_0("loss_valid: {}".format(loss_valid.mean()))
            #print_rank_0("l2_loss: {}".format(l2_loss))
            loss = loss_valid.mean() + l2_loss_valid.mean()
            zero_loss2 = torch.zeros_like(loss)
            loss = torch.where(torch.isnan(loss), zero_loss2, loss)
            #print_rank_0("final_loss: {}".format(loss))

            total_loss.append(loss)

        total_loss = torch.stack(total_loss)
        if self.reduction == "mean":
            total_loss = total_loss.mean()
        #print_rank_0("total_loss: {}".format(total_loss))
        return total_loss
