from skimage.feature import peak_local_max
import scipy.signal
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from skimage.morphology import convex_hull_image
import numpy as np
import skimage.color as skc
import os
import matplotlib as mpl

DEBUG=False

def mkdir_if_not_exist(inputdir):
    if not os.path.exists(inputdir):
        os.makedirs(inputdir)
    return inputdir


def autocorr_image(imgray,mode='full'): 
    imgray_ = imgray - np.mean(imgray) # center the data
    return scipy.signal.fftconvolve(imgray_, imgray_[::-1,::-1], mode=mode) # calculate the autocorrelation 


def GetCropRatio(imgray,min_distance='infer'):
    '''
    This function receives a grayscale image as input and finds
    the ratio on x-axis to be cropped to remove similar samples within 
    the same slide. This is done by calculating the 2d autocorrelation 
    and finding the distance between peaks.
    Right now the cropping is only done assuming that the samples are
    placed horizontally
    '''
    if min_distance=='infer':
        min_distance = int(min(imgray.shape)/4)
    imcorr = autocorr_image(imgray)
    peak_x = peak_local_max(imcorr, min_distance=min_distance)[:,1]
    
    if len(peak_x)<=1:
        CropRatio = 1.
    else:
        peak_x.sort()
        CropRatio = (peak_x[1]-peak_x[0])/imgray.shape[1]

    return CropRatio    


def CropSlideCoordRatiosV3(Slide,thumbnail_size = (200,200),
                           min_distance='infer', SaveToFolder = None,
                           ShowIntermediateStates = False):  # remove extra replicates and crop to image

    
    thumb = np.asarray(Slide.get_thumbnail(thumbnail_size)) # create thumbnail (benefits: smaller, less computation and memory)
    imgray = 1-skc.rgb2gray(thumb) # convert to gray scale
    segmentation = imgray>threshold_otsu(imgray) # threshold according to otsu's method
    segmentation_closed = ndi.morphology.binary_closing(segmentation,iterations=5) # do morphological closing on segmentation to remove small details    
    segmentation_masked = segmentation & segmentation_closed # this worked in removing narrow image margins which were different from background.
    segregions = segmentation.mean(axis=0)>0
    idx_crop = np.where(np.diff(segregions))[0][1]+1
    imgray_cropped = imgray[:,:idx_crop] # crop leftmost sample in a slide
    segmentation_cropped = imgray_cropped > threshold_otsu(imgray) # threshold according to otsu's method
    segmentation_cropped_fillholes = ndi.binary_fill_holes(segmentation_cropped) # fill holes
    convhull = convex_hull_image(segmentation_cropped_fillholes) # find convex hull
    idx0 = np.where(np.any(convhull,axis=1))[0][[0,-1]]  # find vertical bounds for cropping background
    idx1 = np.where(np.any(convhull,axis=0))[0][[0,-1]]  # find horizontal bounds for cropping background
    
    # Produce plots and/or save them
    assert (SaveToFolder==None) or (type(SaveToFolder)==str), 'Proper save folder not provided.'     
    if ShowIntermediateStates or (SaveToFolder!=None): 
        _,ax = pl.subplots(4,3,figsize = (15,11));pl.axis('off');
        TitleFontSize = 18
        fig = pl.sca(ax[0,0]);pl.imshow(thumb);pl.axis('off');pl.title('thumbnail',fontsize=TitleFontSize)
        pl.sca(ax[0,1]);pl.imshow(imgray,cmap='gray');pl.axis('off');pl.title('grayscale',fontsize=TitleFontSize)
        pl.sca(ax[0,2]);pl.imshow(segmentation,cmap='gray');pl.axis('off');pl.title('binarize',fontsize=TitleFontSize)
        pl.sca(ax[1,0]);pl.imshow(segmentation_closed,cmap='gray');pl.axis('off');pl.title('mask = 5x morph. closing',fontsize=TitleFontSize)
        pl.sca(ax[1,1]);pl.imshow(segmentation_masked,cmap='gray');pl.axis('off');pl.title('masking',fontsize=TitleFontSize)
        pl.sca(ax[1,2]);pl.plot(segmentation.mean(axis=0));pl.plot(segregions);pl.title('crop regions',fontsize=TitleFontSize)
        pl.sca(ax[2,0]);pl.imshow(segmentation_cropped,cmap='gray');pl.axis('off');pl.title('crop any extra samples',fontsize=TitleFontSize)
        pl.sca(ax[2,1]);pl.imshow(segmentation_cropped_fillholes,cmap='gray');pl.axis('off');pl.title('fill holes',fontsize=TitleFontSize)
        pl.sca(ax[2,2]);pl.imshow(convhull,cmap='gray');pl.axis('off');pl.title('find convex hull',fontsize=TitleFontSize)
        pl.sca(ax[3,0]);pl.imshow(thumb[idx0[0]:idx0[1],idx1[0]:idx1[1]]);pl.axis('off');pl.title('final thumbnail',fontsize=TitleFontSize)
        pl.sca(ax[3,1]);pl.axis('off');
        pl.tight_layout()
        
        if SaveToFolder!=None:
            os.makedirs(SaveToFolder)
            pl.savefig(os.path.join(SaveToFolder,'preprocess.jpg'), 
                       bbox_inches='tight', pad_inches=0)
            
            if ~ShowIntermediateStates:
                pl.close(pl.gcf())

    return {'v':idx0/thumb.shape[0],'h':idx1/thumb.shape[1]} # return the bounds as ratios


def CropSlideCoordRatiosV4(Slide,thumbnail_size = (200,200),
                           min_distance='infer', SaveToFolder = None,
                           ShowIntermediateStates = False):  # remove extra replicates and crop to image
    
    assert (SaveToFolder==None) or (type(SaveToFolder)==str), 'Proper save folder not provided.'     
    if ShowIntermediateStates or (SaveToFolder!=None): 
        if ShowIntermediateStates:
            mpl.use('Agg')
        else:
            mpl.use('pdf')
    from skimage import measure

    thumb = np.asarray(Slide.get_thumbnail(thumbnail_size)) # create thumbnail (benefits: smaller, less computation and memory)
    imgray = 1-skc.rgb2grey(thumb) # convert to gray scale
    segmentation = imgray>threshold_otsu(imgray) # threshold according to otsu's method
    seg_filled = ndi.binary_fill_holes(segmentation) # fill in the holes
    seg_filled_open = ndi.morphology.binary_opening(seg_filled) # morphologically open the image (remove small spots)
    all_labels = measure.label(seg_filled_open, background=0) # label image regions
    largest_label = np.argmax([region.area for region in measure.regionprops(all_labels)])+1 # find the largest region
    largest_segment = (all_labels == largest_label) # create a mask from largest region
    convhull = convex_hull_image(largest_segment) # find convex hull
    idx0 = np.where(np.any(convhull,axis=1))[0][[0,-1]]  # find vertical bounds for cropping background
    idx1 = np.where(np.any(convhull,axis=0))[0][[0,-1]]  # find horizontal bounds for cropping background

    # Produce plots and/or save them
    if ShowIntermediateStates or (SaveToFolder!=None): 
        import matplotlib.pyplot as pl       

        _,ax = pl.subplots(3,3,figsize = (15,11));pl.axis('off');
        TitleFontSize = 18
        fig = pl.sca(ax[0,0]);pl.imshow(thumb);pl.axis('off');pl.title('thumbnail',fontsize=TitleFontSize)
        pl.sca(ax[0,1]);pl.imshow(imgray,cmap='gray');pl.axis('off');pl.title('grayscale',fontsize=TitleFontSize)
        pl.sca(ax[0,2]);pl.imshow(segmentation,cmap='gray');pl.axis('off');pl.title('binarize',fontsize=TitleFontSize)   
        pl.sca(ax[1,0]);pl.imshow(seg_filled,cmap='gray');pl.axis('off');pl.title('fill the holes',fontsize=TitleFontSize)
        pl.sca(ax[1,1]);pl.imshow(seg_filled_open,cmap='gray');pl.axis('off');pl.title('morphologically open',fontsize=TitleFontSize)
        pl.sca(ax[1,2]);pl.imshow(largest_segment,cmap='gray');pl.axis('off');pl.title('find largest region',fontsize=TitleFontSize)
        pl.sca(ax[2,0]);pl.imshow(convhull,cmap='gray');pl.axis('off');pl.title('find convex hull',fontsize=TitleFontSize)
        pl.sca(ax[2,1]);pl.imshow(thumb[idx0[0]:idx0[1],idx1[0]:idx1[1]]);pl.axis('off');pl.title('cropped thumbnail',fontsize=TitleFontSize)
        pl.sca(ax[2,2]);pl.axis('off');
        pl.tight_layout()
        if SaveToFolder!=None:
            mkdir_if_not_exist(SaveToFolder)
            pl.savefig(os.path.join(SaveToFolder,'preprocess.jpg'), 
                       bbox_inches='tight', pad_inches=0)

            if ~ShowIntermediateStates:
                pl.close(pl.gcf())

    return {'v':idx0/thumb.shape[0],'h':idx1/thumb.shape[1]} # return the bounds as ratios

def CropSlideCoordRatiosLatest(Slide,thumbnail_size = (200,200),
                               min_distance='infer', SaveToFolder = None,
                               ShowIntermediateStates = False):  # remove extra replicates and crop to image
    return CropSlideCoordRatiosV4(Slide,thumbnail_size = thumbnail_size,
                                  min_distance=min_distance, SaveToFolder = SaveToFolder,
                                  ShowIntermediateStates = ShowIntermediateStates)


def ShowCroppedSlide(Slide , ShowIntermediateStates = False, SaveToFolder = None):
    coordratios = CropSlideCoordRatiosLatest(Slide , ShowIntermediateStates=ShowIntermediateStates, SaveToFolder =SaveToFolder)
    n = Slide.level_count-1
    xh0 = (coordratios['h']*Slide.level_dimensions[0][0]).astype(int)
    xv0 = (coordratios['v']*Slide.level_dimensions[0][1]).astype(int)

    xh3 = (coordratios['h']*Slide.level_dimensions[n][0]).astype(int)
    xv3 = (coordratios['v']*Slide.level_dimensions[n][1]).astype(int)
#     thumb = np.asarray(Slide.get_thumbnail((200,200)))

    return Slide.read_region((xh0[0],xv0[0]), n, (xh3[1]-xh3[0],xv3[1]-xv3[0]))
