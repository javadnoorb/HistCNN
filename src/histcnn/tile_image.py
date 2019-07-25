import numpy as np
from openslide import deepzoom
from histcnn import (preprocess_image, util)
import openslide as ops  # installed by using docker (docker exec -it [container-id] bash) and then following this : http://openslide.org/download/
import os

def mkdir_if_not_exist(inputdir):
    if not os.path.exists(inputdir):
        os.makedirs(inputdir)
    return inputdir


def TileSVS(svsFile, outdir, SaveToFile = True, tile_size=512, downsample = 1,
            bg_th = 220, max_bg_frac = 0.5, ProgressBarObj=None, **kwargs):    
    '''
    bg_th : intensities above this will be treated as background
    max_bg_frac : tiles with more background pixels than this fraction will not be saved to the disk. 
                  set this to 1 in order to save all tiles. This parameter only matters if SaveToFile=True
    '''
    L = downsample*tile_size
    Slide = ops.OpenSlide(svsFile)

    if SaveToFile:
        tilesdir = os.path.join(outdir, os.path.basename(svsFile), 'tiles')
        mkdir_if_not_exist(tilesdir)
        
        
    Xdims = np.floor((np.array(Slide.dimensions)-1)/L).astype(int)
    if SaveToFile:
        X = None
    else:
        X = np.zeros((Xdims[0], Xdims[1], tile_side, tile_side, 3), dtype='uint8')


    num_tiles = np.prod(Xdims)
    Pbar = util.ProgressBar(num_tiles, step=1)

    for m in range(Xdims[0]):
        for n in range(Xdims[1]):
            Pbar.NestedUpdate('(%d/%d,%d/%d)'%(m, Xdims[0], n, Xdims[1]), ProgressBarObj=ProgressBarObj)   

            try:
                tile = Slide.read_region((m*L, n*L), 0, (L, L))
            except ops.lowlevel.OpenSlideError:
                print('\n Skipping OpenSlideError, (m, n, L)=({:d}, {:d}, {:d}), {:s}'.format(m, n, L, svsFile))
                Slide = ops.OpenSlide(svsFile)
                tiles = ops.deepzoom.DeepZoomGenerator(Slide, tile_size=tile_size, overlap=0, limit_bounds=False)
                continue

            tile = tile.convert('RGB') # remove alpha channel
            tile = tile.resize((tile_size, tile_size))

            if SaveToFile:
                if (np.array(tile).min(axis=2)>=bg_th).mean() <= max_bg_frac: # only save tiles that are not background
                    outfile = os.path.join(tilesdir,'tile_%d_%d.jpg'%(m,n))
                    tile.save(outfile, "JPEG", quality=100)
            else:
                tile = np.array(tile)
                X[m,n,:tile.shape[0],:tile.shape[1],:] = tile
                
    Slide.close()
    return X

def TileSVS_and_crop_using_deepzoom(svsFile, outdir, SaveToFile = True, tile_size=512, tile_level_step_down = 1, 
            bg_th = 220, max_bg_frac = 0.5, ProgressBarObj=None, no_cropping = False):    
    '''
    bg_th : intensities above this will be treated as background
    max_bg_frac : tiles with more background pixels than this fraction will not be saved to the disk. 
                  set this to 1 in order to save all tiles. This parameter only matters if SaveToFile=True
    '''

    Slide = ops.OpenSlide(svsFile)
    tiles = ops.deepzoom.DeepZoomGenerator(Slide, tile_size=tile_size, overlap=0, limit_bounds=False)
    assert (np.round(np.array(tiles.level_dimensions[-1])/ np.array(tiles.level_dimensions[-2])) == 2).all(), \
            'level_dimension[-2] should be almost twice smaller than level_dimension[-1] for proper conversion between 20x<->40x'

    if SaveToFile:
        tilesdir = os.path.join(outdir,os.path.basename(svsFile),'tiles')
        mkdir_if_not_exist(tilesdir)
        SaveToFolder = os.path.join(outdir,os.path.basename(svsFile),'preprocessing')
    else:
        SaveToFolder = None

    tile_level = tiles.level_count-tile_level_step_down
    tile_side = np.array(tiles.get_tile(tile_level, (0,0))).shape
    assert tile_side[0]==tile_side[1]
    tile_side = tile_side[0]
    tiles_size = tiles.level_tiles[tile_level]

    if no_cropping:
        coordratios = {'h': np.array([0., 1.]), 'v': np.array([0., 1.])}
    else:
        coordratios = preprocess_image.CropSlideCoordRatiosLatest(Slide, thumbnail_size = (200,200),
                                                                  min_distance='infer', 
                                                                  SaveToFolder = SaveToFolder,
                                                                  ShowIntermediateStates = False)

    xh = (coordratios['h']*tiles_size[0]).astype(int)
    xv = (coordratios['v']*tiles_size[1]).astype(int)

    Xdims = (xh[1]-xh[0],xv[1]-xv[0])
    if SaveToFile:
        X = None
    else:
        X = np.zeros((Xdims[0], Xdims[1], tile_side, tile_side, 3), dtype='uint8')

    num_tiles = np.prod(Xdims)
    Pbar = util.ProgressBar(num_tiles,step=1)

    for m in range(Xdims[0]):
        for n in range(Xdims[1]):
            Pbar.NestedUpdate('(%d/%d,%d/%d)'%(m,Xdims[0],n,Xdims[1]),ProgressBarObj=ProgressBarObj)   
            tile = tiles.get_tile(tile_level, (m+xh[0],n+xv[0]))

            if SaveToFile:
                if (np.array(tile).min(axis=2)>=bg_th).mean() <= max_bg_frac: # only save tiles that are not background
                    outfile = os.path.join(tilesdir,'tile_%d_%d.jpg'%(m,n))
                    tile.save(outfile, "JPEG", quality=100)
            else:
                tile = np.array(tile)
                X[m,n,:tile.shape[0],:tile.shape[1],:] = tile
                
    Slide.close()
    return X


