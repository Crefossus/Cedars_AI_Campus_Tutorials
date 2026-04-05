import matplotlib
matplotlib.rc('image', cmap='gray')
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
import cv2
import seaborn as sns
from skimage.morphology import binary_opening,binary_closing,disk
from sklearn.metrics import f1_score, classification_report, confusion_matrix, mean_absolute_error as mean_absolute_error_

def mean_absolute_error(y_true,y_pred):
    return mean_absolute_error_(y_true[~np.isnan(y_pred)],y_pred[~np.isnan(y_pred)])

def postprocess_mask(mask,opening_size=5,closing_size=8):
    mask=binary_opening(mask,disk(opening_size))
    mask=binary_closing(mask,disk(closing_size))
    return mask

def predict_nucleus_cytoplasm_masks(img_gray,nucleus_threshold=0.4,cytoplasm_threshold=0.6,postprocess=False,return_pred=False):
    nucleus_mask=img_gray<=nucleus_threshold
    if postprocess: nucleus_mask=postprocess_mask(nucleus_mask)

    cytoplasm_mask=img_gray<=cytoplasm_threshold
    if postprocess: cytoplasm_mask=postprocess_mask(cytoplasm_mask)
    cytoplasm_mask=np.logical_and(cytoplasm_mask,~nucleus_mask)

    y_pred=2*nucleus_mask.astype(int)+cytoplasm_mask.astype(int)
    if return_pred: return nucleus_mask,cytoplasm_mask,y_pred
    return nucleus_mask,cytoplasm_mask

def calc_nc_ratio_from_image(img_gray,nucleus_threshold=0.4,cytoplasm_threshold=0.6,postprocess=False):
    nucleus_mask,cytoplasm_mask=predict_nucleus_cytoplasm_masks(img_gray,nucleus_threshold,cytoplasm_threshold,postprocess=postprocess)

    nucleus_area=np.sum(nucleus_mask)
    cytoplasm_area=np.sum(cytoplasm_mask)
    return nucleus_area/(nucleus_area+cytoplasm_area)

def get_nc_ratio_from_mask(y_mask):
    nucleus_mask=y_mask==2
    cytoplasm_mask=y_mask==1
    nucleus_area=np.sum(nucleus_mask)
    cytoplasm_area=np.sum(cytoplasm_mask)
    return nucleus_area/(nucleus_area+cytoplasm_area)

def get_nucleus_cytoplasm_masks(y_true):
    return y_true==2,y_true==1

def calculate_true_NC_ratios(Y_train,Y_val,Y_test):
    nucleus_train_true=[]
    for seg_mask in Y_train:
        nucleus_train_true.append((seg_mask==2).sum())
    nucleus_train_true=np.array(nucleus_train_true)
    nucleus_train_true=(Y_train.reshape(Y_train.shape[0],-1)==2).sum(axis=1)
    cytoplasm_train_true=(Y_train.reshape(Y_train.shape[0],-1)==1).sum(axis=1)
    nc_ratio_train_true=nucleus_train_true/(nucleus_train_true+cytoplasm_train_true)
    
    nucleus_val_true=(Y_val.reshape(Y_val.shape[0],-1)==2).sum(axis=1)
    cytoplasm_val_true=(Y_val.reshape(Y_val.shape[0],-1)==1).sum(axis=1)
    nc_ratio_val_true=nucleus_val_true/(nucleus_val_true+cytoplasm_val_true)
    
    nucleus_test_true=(Y_test.reshape(Y_test.shape[0],-1)==2).sum(axis=1)
    cytoplasm_test_true=(Y_test.reshape(Y_test.shape[0],-1)==1).sum(axis=1)
    nc_ratio_test_true=nucleus_test_true/(nucleus_test_true+cytoplasm_test_true)
    return nc_ratio_train_true,nc_ratio_val_true,nc_ratio_test_true

def plot_pred_vs_gt(cytoplasm_mask, nucleus_mask,
                    cytoplasm_gt, nucleus_gt):
    # Predicted
    plt.figure()
    plt.subplot(121)
    plt.imshow(cytoplasm_mask, cmap="Greens")
    plt.title("Cytoplasm")

    plt.subplot(122)
    plt.imshow(nucleus_mask, cmap="Blues")
    plt.title("Nucleus")

    plt.suptitle("Predicted", y=0.8)
    plt.show()

    # Ground truth
    plt.figure()
    plt.subplot(121)
    plt.imshow(cytoplasm_gt, cmap="Greens")
    plt.title("Cytoplasm")

    plt.subplot(122)
    plt.imshow(nucleus_gt, cmap="Blues")
    plt.title("Nucleus")

    plt.suptitle("True / Annotated", y=0.8)
    plt.show()

def plot_segmentation_breakdown(cytoplasm_mask, nucleus_mask,
                                cytoplasm_gt, nucleus_gt):

    plt.figure(figsize=(14, 7))

    # --- Cytoplasm ---
    cyto_tp = cytoplasm_mask & cytoplasm_gt
    cyto_fp = cytoplasm_mask & ~cytoplasm_gt
    cyto_fn = ~cytoplasm_mask & cytoplasm_gt

    plt.subplot(241)
    plt.imshow(cytoplasm_gt, cmap="Greens")
    plt.title("Cyto GT")
    plt.axis("off")

    plt.subplot(242)
    plt.imshow(cytoplasm_mask, cmap="Greens")
    plt.title("Cyto Pred")
    plt.axis("off")

    plt.subplot(243)
    plt.imshow(cyto_tp, cmap="Greens")
    plt.title("Cyto TP")
    plt.axis("off")

    cyto_error = np.zeros((*cytoplasm_mask.shape, 3))
    cyto_error[cyto_fp] = [1, 0, 0]
    cyto_error[cyto_fn] = [0, 0, 1]

    plt.subplot(244)
    plt.imshow(cyto_error)
    plt.title("Cyto Errors (R=FP, B=FN)")
    plt.axis("off")

    # --- Nucleus ---
    nuc_tp = nucleus_mask & nucleus_gt
    nuc_fp = nucleus_mask & ~nucleus_gt
    nuc_fn = ~nucleus_mask & nucleus_gt

    plt.subplot(245)
    plt.imshow(nucleus_gt, cmap="Blues")
    plt.title("Nucleus GT")
    plt.axis("off")

    plt.subplot(246)
    plt.imshow(nucleus_mask, cmap="Blues")
    plt.title("Nucleus Pred")
    plt.axis("off")

    plt.subplot(247)
    plt.imshow(nuc_tp, cmap="Blues")
    plt.title("Nucleus TP")
    plt.axis("off")

    nuc_error = np.zeros((*nucleus_mask.shape, 3))
    nuc_error[nuc_fp] = [1, 0, 0]
    nuc_error[nuc_fn] = [0, 0, 1]

    plt.subplot(248)
    plt.imshow(nuc_error)
    plt.title("Nucleus Errors (R=FP, B=FN)")
    plt.axis("off")

    plt.suptitle("Segmentation Breakdown by Class")
    plt.tight_layout()
    plt.show()

def plot_confusion_analysis(y_true, y_pred):
    # --- Full 3x3 CM ---
    cm = confusion_matrix(
        y_true.flatten(),
        y_pred.flatten(),
        labels=[0, 1, 2]
    )

    # --- Binary masks ---
    cyto_true = (y_true == 1).astype(int)
    cyto_pred = (y_pred == 1).astype(int)

    nuc_true = (y_true == 2).astype(int)
    nuc_pred = (y_pred == 2).astype(int)

    # --- 2x2 CMs ---
    cm_cyto = confusion_matrix(cyto_true.flatten(), cyto_pred.flatten(), labels=[0,1])
    cm_nuc  = confusion_matrix(nuc_true.flatten(),  nuc_pred.flatten(),  labels=[0,1])

    plt.figure(figsize=(15, 4))

    # --- 3x3 ---
    plt.subplot(131)
    plt.imshow(cm, cmap="viridis")
    plt.title("Full Confusion Matrix")
    plt.colorbar()

    classes = ["Background", "Cytoplasm", "Nucleus"]
    plt.xticks(range(3), classes, rotation=45)
    plt.yticks(range(3), classes)

    for i in range(3):
        for j in range(3):
            color = "white" if cm[i, j] > cm.max()/2 else "black"
            plt.text(j, i, cm[i, j],
                     ha="center", va="center", color=color)

    plt.ylabel("True")
    plt.xlabel("Predicted")

    # --- Cytoplasm ---
    plt.subplot(132)
    plt.imshow(cm_cyto, cmap="Blues")
    plt.title("Cytoplasm (2×2)")
    plt.colorbar()

    labels = ["Not Cyto", "Cyto"]
    plt.xticks([0,1], labels)
    plt.yticks([0,1], labels)

    names = [["TN", "FP"],
             ["FN", "TP"]]

    for i in range(2):
        for j in range(2):
            val = cm_cyto[i, j]
            color = "white" if val > cm_cyto.max()/2 else "black"
            plt.text(j, i, f"{names[i][j]}\n{val}",
                     ha="center", va="center", color=color)

    plt.ylabel("True")
    plt.xlabel("Predicted")

    # --- Nucleus ---
    plt.subplot(133)
    plt.imshow(cm_nuc, cmap="Purples")
    plt.title("Nucleus (2×2)")
    plt.colorbar()

    labels = ["Not Nucleus", "Nucleus"]
    plt.xticks([0,1], labels)
    plt.yticks([0,1], labels)

    for i in range(2):
        for j in range(2):
            val = cm_nuc[i, j]
            color = "white" if val > cm_nuc.max()/2 else "black"
            plt.text(j, i, f"{names[i][j]}\n{val}",
                     ha="center", va="center", color=color)

    plt.ylabel("True")
    plt.xlabel("Predicted")

    plt.suptitle("Segmentation Confusion Analysis", y=1.05)
    plt.tight_layout()
    plt.show()