from google.cloud import storage
from google.cloud.storage import Blob
import os
from histcnn import (tile_image,
                 preprocess_image,
                 util)
import openslide as ops

DATA_PATH = pkg_resources.resource_filename('histcnn', 'data/')

def mkdir_if_not_exist(inputdir):
    if not os.path.exists(inputdir):
        os.makedirs(inputdir)
    return inputdir

def DownloadSVS(BlobName, outdir='.', project='jax-nihcc-res-00-0011'):
#     bucketname = 'isb-cgc-open'
    bucketname = BlobName[5:].split('/')[0]
    client = storage.Client(project=project)
    bucket = client.get_bucket(bucketname)
    blob = Blob(BlobName.lstrip('gs://' + bucketname + '/'), bucket)
    with open(os.path.join(mkdir_if_not_exist(outdir),BlobName.split('/')[-1]), 'wb') as file_obj:
        blob.download_to_file(file_obj)


def DownloadSVS_TileIt_DeleteIt(slide_info, work_space, svsdirname='tmp',
                                tile_level_step_down = 1, tile_size=512,
                                bg_th = 220 , max_bg_frac = 0.5, downsample = 1,
                                ProgressBarObj=None, no_cropping = False,
                                project='jax-nihcc-res-00-0011'):
    '''
    This function checks if preprocessing file exists.
    If not it downloads the svs files,
    applies preprocesing (mainly cropping), and
    tiles the cropped image, saves the tiles,
    and then deletes the svs
    '''
    svsdir = os.path.join(work_space,svsdirname)
    svsFile = os.path.join(svsdir,slide_info['svsFilename'])
    if not os.path.isfile(os.path.join(work_space,
                                       os.path.basename(svsFile),
                                       'preprocessing/preprocess.jpg')):
        DownloadSVS(slide_info['GCSurl'], outdir=svsdir, project=project)
        tile_image.TileSVS(svsFile, work_space, SaveToFile = True, tile_size=tile_size,
                           tile_level_step_down = tile_level_step_down,
                           bg_th = bg_th , max_bg_frac = max_bg_frac, downsample = downsample,
                           ProgressBarObj=ProgressBarObj, no_cropping = no_cropping)
        os.remove(svsFile)

def DownloadSVS_Preprocess_DeleteIt(slide_info, work_space, svsdirname='tmp', tile_level_step_down = 2, tile_size=190):
    svsdir = os.path.join(work_space, svsdirname)
    svsFile = os.path.join(svsdir, slide_info['svsFilename'])
    SaveToFolder = os.path.join(work_space, os.path.basename(svsFile), 'preprocessing')
    if not os.path.isfile(os.path.join(SaveToFolder, 'preprocess.jpg')):
        DownloadSVS(slide_info['GCSurl'], outdir=svsdir)
        Slide = ops.OpenSlide(svsFile)
        preprocess_image.ShowCroppedSlide(Slide, ShowIntermediateStates = False, SaveToFolder=SaveToFolder)
        Slide.close()
        os.remove(svsFile)

def maybe_tile_svs_files_and_save_them(df, output_path, downsample = 1, no_cropping = False):
    Pbar = util.ProgressBar(len(df))
    df_shuffle = df.sample(frac=1,random_state = 0)
    for _,slide_info in df_shuffle.iterrows():
        Pbar.Update(slide_info['slide_barcode'])
        try:
            DownloadSVS_TileIt_DeleteIt(slide_info, output_path,
                                      tile_level_step_down = 2, tile_size = 190,
                                      bg_th = 220 , max_bg_frac = 0.5, downsample = downsample,
                                      ProgressBarObj=Pbar, no_cropping = no_cropping)
        except ops.OpenSlideError as opserr:
            print('\n OpenSlideError:',opserr,
                  '. This seems to be due to corrupted metadata regarding level sizes.\n Filename: ',
                  slide_info['svsFilename']
                  )

    print('\n')
