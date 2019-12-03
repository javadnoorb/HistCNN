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
import matplotlib.backends.backend_pdf
from PIL import Image
from tqdm import tqdm

DATA_PATH = pkg_resources.resource_filename('histcnn', 'data/')

def save_all_overlaid_tumormaps(cancertype1, cancertype2, 
                                picklefile, svs_download_dir,
                                L=30000, figsize=(15, 15), 
                                single_output=True, 
                                number_of_slides = 5, 
                                downsample = 2, tile_size = 512, wsi=False):

    votes, predictions_df = get_predictions_df(cancertype1, cancertype2, picklefile)
    slides = get_best_slides(votes, predictions_df, number_of_slides = number_of_slides)

    for slide_id in tqdm(slides):
        try:
            save_overlaid_tumormap(slide_id, svs_download_dir, 
                                       cancertype1, cancertype2, 
                                       votes, predictions_df,
                                       L=L, figsize=figsize, 
                                       single_output=single_output, wsi=wsi)
        except Exception as exc:
            print(slide_id)
#             print(traceback.format_exc())
            print(exc)
            
        plt.close('all')

def save_overlaid_tumormap(slide_id, svs_download_dir, 
                           cancertype1, cancertype2, 
                           votes, predictions_df,
                           L=10000, figsize=(15, 15),
                           single_output=True, 
                           tile_size = 512, binary=True, wsi=False):
    
    metadata = pd.read_csv(os.path.join(DATA_PATH, 'TCGA_slide_images_metadata.txt'))
    AppMag = metadata[metadata['slide_barcode'] == slide_id]['AppMag'].iloc[0]

    
    svsFile = download_slide(slide_id, svs_download_dir, force_download=False)
    slide = ops.OpenSlide(svsFile)

    figtitle, outputfile = get_figtitle(cancertype1, cancertype2, votes, predictions_df, slide_id)

    tumormap = get_tumormap(predictions_df, slide, slide_id, 
                            downsample = AppMag, 
                            tile_size = tile_size, binary=binary)
    if wsi:
        level = 0 
        thumb = slide.read_region((0, 0), level, slide.level_dimensions[level])
        thumb = np.asarray(thumb)
    else:
        thumb = get_thumbnail(slide, L=L, show_figure=False)
    
    if not single_output:
        pdf = matplotlib.backends.backend_pdf.PdfPages(outputfile)
    thumb_pil = overlay_maps(thumb, slide, tumormap, figsize=figsize, plot_output=single_output)
    
    if single_output:
        thumb_pil.save(outputfile.rstrip('pdf')+'png')
    else:
        plt.figure(figsize=figsize)
        plt.imshow(thumb_pil)
        plt.title(figtitle, fontsize=14, fontweight='bold')
        plt.axis('off');
        
        pdf.savefig(bbox_inches='tight')
        plt.figure(figsize=figsize)
        plt.imshow(thumb)
        plt.axis('off');
        pdf.savefig(bbox_inches='tight')

        pdf.close()
        
            
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
        image_files_metadata, imagefilenames, predictions_list, label_names, 
        final_softmax_outputs_list=final_softmax_outputs_list)

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

def get_tumor_map_cropped(predictions_df, slide_id, binary=True):
    label = 'label'
    tmp = predictions_df[predictions_df['sample_id'] == slide_id].copy()
    pattern = re.compile('(\d+)_(\d+)\.jpg$')
    tmp['re_groups'] = tmp['image_filename'].map(lambda x: re.search(pattern, x))
    tmp['coord_x'] = tmp['re_groups'].map(lambda x: int(x.group(1)))
    tmp['coord_y'] = tmp['re_groups'].map(lambda x: int(x.group(2)))
    if binary:
        tmp = pd.pivot(tmp, index='coord_x', columns='coord_y', values=label+'_pred')
        tmp = (tmp + 1).fillna(0).astype(int)
    else:
        tmp['pred_probs'] = tmp['pred_probs'].map(lambda x: x[1])
        tmp = pd.pivot(tmp, index='coord_x', columns='coord_y', values='pred_probs')  
        tmp = (tmp + 1).fillna(0)
    return tmp

def resize_tumormap(tumormap, scale=1):
    tumormap_resized = sktr.rescale(tumormap, scale, anti_aliasing=False, order=0, preserve_range=True, multichannel=True)
    return tumormap_resized.astype('uint8')

def overlay_maps(thumb, slide, tumormap, 
                 downsample=2, tile_size=512, 
                 alpha=0.5, figsize=(15, 15), plot_output=True):
    L = downsample*tile_size
    Xdims = np.floor((np.array(slide.dimensions)-1)/L).astype(int)
    floorcorrection = (np.array(slide.dimensions) / (Xdims*L))[::-1]

    size_ratio = np.array(thumb.shape[:2])/np.array(tumormap.shape[:2])
    tumormap_resized = resize_tumormap(tumormap, scale=size_ratio/floorcorrection)
    
    if plot_output:
        fig, ax = plt.subplots(figsize=figsize)

    thumb_pil = Image.fromarray(thumb)
#     thumb_pil.putalpha(255)
    tumormap_resized = Image.fromarray(tumormap_resized)
    tumormap_resized.putalpha(int(255*alpha))
    thumb_pil.paste(tumormap_resized, (0,0), mask=tumormap_resized)
    
    if plot_output:
        plt.imshow(thumb_pil)
    return thumb_pil

def convert_tumormap_to_rgb(a, cmap = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], immax=255, 
                            binary=True, prob_map_cmap=[0.0, 1.0, 0.0], redblue=True):
    if binary:
        img = np.stack([a]*3, axis=2).astype(int)
        rgb = 0
        for n in range(3):
            rgb+=np.multiply((img == n), cmap[n])
        return rgb * immax
    else:
        assert sum(prob_map_cmap) == 1
        rgb = a - 1
        
        if redblue:
            rgb = np.stack([rgb*(rgb>0.5), rgb*0, rgb*(rgb<=0.5)], axis=2)
        else:
            rgb = np.stack([rgb*prob_map_cmap[0], rgb*prob_map_cmap[1], rgb*prob_map_cmap[2]], axis=2)#.astype(int)
        rgb[a == 0] = 1.0
        return (rgb * immax).astype(int)

def get_tumormap(predictions_df, slide, slide_id, downsample = 2, 
                 tile_size = 512, convert_to_rgb = True,
                 cmap = [[1, 1, 1], [0, 0, 0], [0, 1, 0]],
                 immax=255, binary=True, prob_map_cmap=[0.0, 1.0, 0.0]):

    tumormap = get_tumor_map_cropped(predictions_df, slide_id, binary=binary)
    tumormap = pad_with_zeros(tumormap)
    tumormap = tumormap.T.values
    tumormap = pad_tumormap_rightdown(tumormap, slide, downsample = 2, tile_size = 512)
    if convert_to_rgb:
        tumormap = convert_tumormap_to_rgb(tumormap, cmap=cmap, 
                                           immax=immax, binary=binary, 
                                           prob_map_cmap=prob_map_cmap)
#         tumormap[np.isnan(tumormap)] = immax
    return tumormap

def get_thumbnail(slide, L = 1000, show_figure=True, figsize=(15, 15)):
    th = slide.get_thumbnail((L, L))
    thumb = np.asarray(th)
    
    if show_figure:
        plt.figure(figsize=figsize)
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
    outputfile = 'her2-tumormaps_overlayed/{:s}-{:s}-{:s}-freq{:.2f}-{:s}.tumormap.pdf'.format(
            cancertype1, cancertype2, label.replace(' ', '_'), freq, slide_id)
    
    return figtitle, outputfile