import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import csv
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

#######plotCluster#######
#bar location=bottom + axs
def ind2ij(ind,size,axis):
    i,j=divmod(ind-1,size)
    i+=1
    j+=1
    return np.array([i,j])[axis]

def plot_label_image(a,pred_y,cmp,save=None,mask=None,figsize=(5,5),anno=False,
                     ifshow=True, location = 'bottom', return_mat=False,
                     cbar = True, cticks = True, ax = None):
    
    SZ1 = int(max(a.obsm['spatial'][:,0])) + 1
    SZ2 = int(max(a.obsm['spatial'][:,1])) + 1
    to_labeling_pred_y = np.array(pred_y.astype('int'))
    to_labeling_pred_y_min = to_labeling_pred_y.min()
    
    unique_cls = np.unique(pred_y).shape[0]
    cluster_cmp = cmp.copy()
    
    if mask is not None:
        mask = np.array(mask).astype('int')
        for to_mask in range(unique_cls):
            if to_mask in mask:
                continue
            cluster_cmp[to_mask]='k'
    labeling_plot_cmp = []
    labeling_plot_cmp.extend(cluster_cmp)
    labeling = to_labeling_pred_y

    img1 = labeling.reshape((SZ1,SZ2)).T
    plt.figure(figsize=figsize)

    ticks=np.arange(np.min(img1),np.max(img1)+1)
    boundaries = np.arange(np.min(img1)-0.5,np.max(img1)+1.5)

    sns.heatmap(img1,cmap=labeling_plot_cmp,linewidths=0,linecolor='k',square=True, \
                xticklabels = [''], yticklabels = [''], cbar = cbar, \
                cbar_kws={"ticks": ticks if cticks else [],  "use_gridspec":False, "location":location,\
                          "boundaries":boundaries,'fraction':0.046,'pad':0.04}, ax = ax)

    if ax is None:
        plt.xticks([])
        plt.yticks([])
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if save is not None:
        plt.savefig(save,transparent=False,bbox_inches='tight')

    if anno:
        num_cells = pred_y.shape[0]
        for i in range(num_cells):
            cur_idx = i + 1
            cur_ind = cell_pos[cell_idx==cur_idx][0]
            if to_labeling_pred_y[i]-to_labeling_pred_y_min in mask:
                plt.annotate(str(cur_idx-1),(ind2ij(cur_ind,IMG_SZ1,1), \
                                             ind2ij(cur_ind,IMG_SZ1,0)),color='red')

    if ax is None and ifshow:
        plt.show()
    if return_mat:
        return img1

    
def plotCluster(a,color,groups = None,save = None, location = 'bottom', cbar = True,
                cticks = True, return_mat=False, ax = None):
    if return_mat:
        plot_mat = plot_label_image(a,a.obs[color],a.uns[cls+'_colors'],mask=groups,save=save,
                                    cbar = cbar, cticks = cticks,
                                    location = location, return_mat=return_mat, ax = ax)
        return plot_mat
    else:
        plot_label_image(a,a.obs[color],a.uns[color+'_colors'],mask=groups,
                         cbar = cbar, cticks = cticks,
                         location = location, save=save, ax = ax)


        
#######plot_colortable#######
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_colortable(colors, title, sort_colors=True, emptycols=0):

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    topmargin = 40

    if sort_colors is True:
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                         name)
                        for name, color in colors.items())
        names = [name for hsv, name in by_hsv]
    else:
        names = list(colors)

    n = len(names)
    ncols = 4 - emptycols
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-topmargin)/height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(title, fontsize=24, loc="left", pad=10)

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )
    return fig





#######plotLabelImageOnImg#######
#bar location=bottom + axs
def ind2ij(ind,size,axis):
    i,j=divmod(ind-1,size)
    i+=1
    j+=1
    return np.array([i,j])[axis]

def plotLabelImageOnImg(a, pred_y, cmp, img,  offset = (0,0), save=None,
                        mask=None, figsize=(5,5), anno=False, 
                        ifshow=True, location = 'bottom', return_mat=False, ax = None):
    
    SZ1 = int(max(a.obsm['spatial'][:,0])) + 1
    SZ2 = int(max(a.obsm['spatial'][:,1])) + 1
    import copy
    img = copy.deepcopy(img)
    img = img.resize((SZ1,SZ2))
    pixelMap = img.load()

    to_labeling_pred_y = np.array(pred_y.astype('int'))
    to_labeling_pred_y_min = to_labeling_pred_y.min()
    unique_cls = np.unique(pred_y).shape[0]

    cluster_cmp = [tuple(ImageColor.getcolor(color, "RGBA")) for color in cmp] 
    
    if mask is None:
        mask = list(range(unique_cls))
    mask = np.array(mask).astype('int')

    labeling = to_labeling_pred_y - 1
    labeling = np.flip(labeling.reshape((SZ1,SZ2)),1)#.T

    for i in range(SZ1):
        for j in range(SZ2):
            if i + offset[0] >= SZ1 or j + offset[1] >= SZ2: continue
            if labeling[i,j] in mask:
                pixelMap[i + offset[0] ,j + offset[1]] = cluster_cmp[labeling[i,j]]

    if ax is None:
        plt.imshow(img)
        ymin, ymax = plt.ylim();
        plt.ylim(ymax,ymin)
        
    else:
        ax.imshow(img)
        ax.set_ylim(ax.get_ylim()[::-1])

    labeling = to_labeling_pred_y
    img1 = labeling.reshape((SZ1,SZ2))#.T
    plt.figure(figsize=figsize)
    # plt.imshow(img1)
    ticks=np.arange(np.min(img1),np.max(img1)+1)
    boundaries = np.arange(np.min(img1)-0.5,np.max(img1)+1.5)

    if ax is None:
        plt.xticks([])
        plt.yticks([])
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if save is not None:
        plt.savefig(save,transparent=False,bbox_inches='tight')

    if anno:
        num_cells = pred_y.shape[0]
        for i in range(num_cells):
            cur_idx = i + 1
            cur_ind = cell_pos[cell_idx==cur_idx][0]
            if to_labeling_pred_y[i]-to_labeling_pred_y_min in mask:
                plt.annotate(str(cur_idx-1),(ind2ij(cur_ind,IMG_SZ1,1), \
                                             ind2ij(cur_ind,IMG_SZ1,0)),color='red')

    if ax is None and ifshow:
        plt.show()
    if return_mat:
        return img1

    
def plotClusterOnImg(a, color, img, groups = None, save = None, offset = (0,0), 
                     location = 'bottom', return_mat=False, ax = None):
    if return_mat:
        plot_mat = plotLabelImageOnImg(a,a.obs[color],a.uns[cls+'_colors'], img = img, offset = offset,
                                       mask=groups,save=save, location = location, return_mat=return_mat, ax = ax)
        return plot_mat
    else:
        plotLabelImageOnImg(a,a.obs[color],a.uns[color+'_colors'], img = img,  offset = offset,
                            mask=groups, location = location, save=save, ax = ax)
