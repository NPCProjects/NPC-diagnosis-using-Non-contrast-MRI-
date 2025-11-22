import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import nibabel as nib
import scipy.misc
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from utils import criterions
import tqdm


cudnn.benchmark = True

path = os.path.dirname(__file__)

patch_size = 128


def evaluate_classification(test_loader, model, dataname, feature_mask, mask_name):
    model.is_training = False
    model.eval()

    preds = []
    targets = []
    names = []
    scores_evaluation = []
    all_case_features = []
    prediction_results =[]
    last_layer_features = []

    preds_sne = []
    for data in tqdm.tqdm(test_loader):
        target = data[1]
        x = data[0].cuda()
        name = data[-1]
        names.extend(name)
        if feature_mask is not None:
            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(len(name), 1)
        else:
            mask = data[2]
        mask = mask.cuda()
        _, _, H, W, Z = x.size()
        with torch.no_grad():
            pred, _, _, _, _, _, _, output_features = model(x, mask)
        # pred, _, _, _, _, _, _ = model(x, mask)

        # output_features = output_features.mean(dim=1, keepdim=True)  # hierachial cluster

        # features = {"data": "test", "name": name, "features": output_features.detach().cpu(), "label": target}
        # all_case_features.append(features)
        pred = pred.cpu()


        # pred = model(x, mask)[1].cpu()
        pred_probability = torch.softmax(pred, dim=1)[0][1]

        # pred = F.sigmoid(pred) > 0.5
        pred = pred.argmax(axis=1)


        prediction = {"name": name,  "label": target, "pred":pred, 'pred_probability':pred_probability}
        prediction_results.append(prediction)

        # target = target.argmax(axis=1)
        preds.extend(pred.tolist())
        preds_sne.append(pred)
        targets.extend(target.tolist())
        # last_layer_features.append(output_features.flatten().detach().cpu().numpy().squeeze())

    accuracy = accuracy_score(targets, preds)
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
    recall = recall_score(targets, preds)
    precision = precision_score(targets, preds)
    specificity = tn / (tn + fp)
    f1 = f1_score(targets, preds)

    print('Accuracy', accuracy)
    print('Recall/sensitivity', recall)
    print('Precision', precision)
    print('F1', f1)

    print('specificity', specificity)
    #
    #
    #
    scores_evaluation.append(accuracy)
    # scores_evaluation.append(specificity)
    scores_evaluation.append(recall)
    scores_evaluation.append(f1)
    scores_evaluation = np.array(scores_evaluation)
    scores_evaluation = scores_evaluation.T
    for k, name in enumerate(names):
        msg = 'Subject {}/{}'.format((k+1), len(names))
        msg += '{:>20}, '.format(name)

        msg += 'True Label {}, Prediction {}'.format(targets[k], preds[k])

        # vals_evaluation.update(scores_evaluation[k])
        # msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
        #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])

        # logging.info(msg)
    return accuracy, recall, precision, f1,all_case_features,preds, prediction_results

