import skimage.transform as sktr
import re
import openslide as ops
import pickle
from histcnn import plotting_cnn, util
import os
import pkg_resources
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = pkg_resources.resource_filename('histcnn', 'data/')

def get_best_slides(votes, predictions_df, number_of_slides = 5):
    slides_list_no_amp = votes[votes['label'] == 0]['label_pred'].sort_values(ascending=True).index[:number_of_slides].tolist()
    slides_list_with_amp = votes[votes['label'] == 1]['label_pred'].sort_values(ascending=False).index[:number_of_slides].tolist()
    slides = slides_list_no_amp + slides_list_with_amp
    return slides

def get_predictions_df(cancertype1, cancertype2, picklefile):
    [image_files_metadata, test_accuracies_list, predictions_list,
     confusion_matrices_list, imagefilenames, final_softmax_outputs_list] = pickle.load(open(picklefile, 'rb'))

    image_files_metadata['label_name'].replace({'normal': 'not amplified', 'tumor': 'amplified'}, inplace=True)

    label_names = ['label']
    votes, predictions_df = plotting_cnn.get_per_slide_average_predictions(
        image_files_metadata, imagefilenames, predictions_list, label_names)

    predictions_df.rename(columns={'rel_path': 'image_filename'}, inplace=True)
    predictions_df['image_filename'] = predictions_df['image_filename'].str.rstrip('_cached.txt')
    
    return votes, predictions_df

def pad_index_col_with_zeros(input_df):
    df = input_df.copy()
    indeces = df.index
    remainder_inds = list(set(range(max(indeces))) - set(indeces))

    for idx in remainder_inds:
        df.loc[idx, :] = 0
    df.sort_index(inplace=True)    
    return df

def pad_with_zeros(df):
    return pad_index_col_with_zeros(pad_index_col_with_zeros(df).T).T

def pad_tumormap_rightdown(tumormap, slide, downsample = 2, tile_size = 512):
    L = downsample*tile_size
    Xdims = np.floor((np.array(slide.dimensions)-1)/L).astype(int)
    tumormap_padded = np.pad(tumormap, ((0, Xdims[1]- tumormap.shape[0]), (0, Xdims[0]- tumormap.shape[1])))
    return tumormap_padded

def get_tumor_map_cropped(predictions_df, slide_id):
    label = 'label'
    tmp = predictions_df[predictions_df['sample_id'] == slide_id].copy()
    pattern = re.compile('(\d+)_(\d+)\.jpg$')
    tmp['re_groups'] = tmp['image_filename'].map(lambda x: re.search(pattern, x))
    tmp['coord_x'] = tmp['re_groups'].map(lambda x: int(x.group(1)))
    tmp['coord_y'] = tmp['re_groups'].map(lambda x: int(x.group(2)))
    tmp = pd.pivot(tmp, index='coord_x', columns='coord_y', values=label+'_pred')
    tmp = (tmp + 1).fillna(0).astype(int)
    return tmp

def resize_tumormap(tumormap, scale=1):
    tumormap_resized = sktr.rescale(tumormap, scale, anti_aliasing=False, order=0, preserve_range=True, multichannel=True)
    return tumormap_resized.astype('uint8')

def overlay_maps(thumb, slide, tumormap, downsample=2, tile_size=512):
    L = downsample*tile_size
    Xdims = np.floor((np.array(slide.dimensions)-1)/L).astype(int)
    floorcorrection = (np.array(slide.dimensions) / (Xdims*L))[::-1]

    size_ratio = np.array(thumb.shape[:2])/np.array(tumormap.shape[:2])
    tumormap_resized = resize_tumormap(tumormap, scale=size_ratio/floorcorrection)

    fig, ax = plt.subplots(figsize=(15, 15))

    ax.imshow(thumb)
    ax.imshow(tumormap_resized, alpha=0.2)

def convert_tumormap_to_rgb(a, cmap = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], immax=255):
    img = np.stack([a]*3, axis=2).astype(int)
    
    rgb = 0
    for n in range(3):
        rgb+=np.multiply((img == n), cmap[n])
    return rgb * immax

def get_tumormap(predictions_df, slide, slide_id, downsample = 2, 
                 tile_size = 512, convert_to_rgb = True,
                 cmap = [[1, 1, 1], [0, 0, 0], [0, 1, 0]],
                 immax=255):
    tumormap = get_tumor_map_cropped(predictions_df, slide_id)
    tumormap = pad_with_zeros(tumormap)
    tumormap = tumormap.T.values
    tumormap = pad_tumormap_rightdown(tumormap, slide, downsample = 2, tile_size = 512)
    if convert_to_rgb:
        tumormap = convert_tumormap_to_rgb(tumormap, cmap=cmap, immax=immax)
    return tumormap

def get_thumbnail(slide, L = 1000, show_figure=True):
    th = slide.get_thumbnail((L, L))
    thumb = np.asarray(th)
    
    if show_figure:
        plt.figure(figsize=(15, 15))
        plt.imshow(thumb)
    return thumb

def download_slide(slide_id, output_dir, force_download=False):
    updated_gcsurls_filename = os.path.join(DATA_PATH, 'TCGA_slide_images_updated_gcsurls.txt')
    updated_gcsurls = pd.read_csv(updated_gcsurls_filename, index_col=1)['file_gcs_url']
    slide_gcsurl = updated_gcsurls.loc[slide_id]
    svsFile = os.path.join(output_dir, slide_gcsurl.split('/')[-1])
    if (not os.path.isfile(svsFile)) | force_download:
        util.gsutil_cp(slide_gcsurl, output_dir, make_dir=True, verbose=True);
    return svsFile

def get_figtitle(cancertype1, cancertype2, votes, predictions_df, slide_id, 
                label_texts = ['her2 not amplified', 'her2 amplified']):
    
    label = label_texts[int(votes.loc[slide_id, 'label'])]
    freq = votes.loc[slide_id, 'label_pred']

    figtitle = '{:s}\nground truth: {:s}\ntrained on: {:s}\ntested on: {:s}\npredicted frequency: {:.2f}'.format(
        slide_id, label, cancertype1.upper(), cancertype2.upper(), freq)
    
    return figtitle