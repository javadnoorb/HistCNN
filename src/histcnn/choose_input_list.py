import glob
import hashlib
import json
from tensorflow.python.util import compat
import pandas as pd
from tqdm._tqdm_notebook import tqdm_notebook
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from google.cloud import storage
import subprocess
from histcnn import util
import pkg_resources

DATA_PATH = pkg_resources.resource_filename('histcnn', 'data/')

def annotate_subtypes(cache_df, tissue,
                      filename = DATA_PATH+'/histotypes.txt', verbose=True):
    if verbose:
        print('Loading histological subtype metadata for each patient')

    
    histotypes = pd.read_csv(filename, sep='\t', index_col=0)

    if verbose:
        print('Merging basic cache annotations with histological types')
    cache_df = cache_df.merge(histotypes, left_on='patient_id', right_index=True)

    print('Labeling the samples...')
    histotypes_counts = load_histotype_guide()
    # filter down to cancer types of interest to avoid possible (rare) subtype naming collision 
    histotypes_counts = histotypes_counts[histotypes_counts['tissue']==tissue]
    histotypes_counts = histotypes_counts[['histological_type', 'label']]
    assert not histotypes_counts['histological_type'].duplicated().any(), 'histological subtypes need to be unique'
    output = cache_df.merge(histotypes_counts, on='histological_type', how='inner')
    output.rename(columns={'label': 'label_name'}, inplace=True)
    output['label'] = pd.factorize(output['label_name'], sort=True)[0]
    if verbose:
        print('Shuffling rows of the resulting dataframe')
    output = output.sample(frac=1).reset_index(drop=True)

    return output

def get_her2_metadata():
    her2cna_files = os.path.join(DATA_PATH, 'her2-cna.txt')
    her2cnas = pd.read_csv(her2cna_files, sep='\t')
    her2cnas = her2cnas[her2cnas['SAMPLE_ID'].map(lambda x: x[-2:]) == '01']
    her2cnas['SAMPLE_ID'] = her2cnas['SAMPLE_ID'].map(lambda x: x[:-3])
    her2cnas.rename(columns = {'SAMPLE_ID': 'patient_id', 'ERBB2': 'cnv'}, inplace=True)
    # her2cnas['cancertype'] = her2cnas['STUDY_ID'].map(lambda x: x.split('_')[0])
    her2cnas = her2cnas[['patient_id',	'cnv']]
    her2cnas.reset_index(drop=True, inplace=True)
    her2cnas['cnv'] = (her2cnas['cnv']>0).astype(int)
    
    return her2cnas

def load_histotype_guide(
    histotypes_counts_file = DATA_PATH+'/histotypes_counts_annotated.xlsx',
    sheet_number=2):
    usecols = ['cancertype', 'histological_type', 'cancertype_fullname', 
               'label', 'sample_counts', 'tissue']
    histotypes_counts = pd.read_excel(histotypes_counts_file, sheet_name=sheet_number)
    histotypes_counts = histotypes_counts[usecols]
    histotypes_counts.dropna(axis=0, inplace=True)
    histotypes_counts['sample_counts'] = histotypes_counts['sample_counts'].astype(int)
    return histotypes_counts

def get_file_list_for_subtypes(tissue,
    caches_gcs_path_list_path = '/sdata/data/pancancer_annotations/individual_cancers/',
    histotypes_counts_file = DATA_PATH+'/histotypes_counts_annotated.xlsx',
    sheet_number=2, verbose=False):
    
    if verbose:
        print('Loading histological subtype guide file')
    histotypes_counts = load_histotype_guide(histotypes_counts_file, sheet_number=sheet_number)

    if verbose:
        print('Loading cache annotations')

    histotypes_counts_filtered = histotypes_counts[histotypes_counts['tissue'] == tissue]

    cancertypes = histotypes_counts_filtered['cancertype'].drop_duplicates().str.lower().tolist()
    meta_tile_anns = None
    for cancertype in tqdm(cancertypes):
        ann_file = os.path.join(caches_gcs_path_list_path ,'{:s}/caches_gcs_path_list.txt'.format(cancertype))
        tile_anns = pd.read_csv(ann_file, sep=',', header = None)
        tile_anns.columns = ['GCSurl']
        meta_tile_anns = pd.concat([meta_tile_anns, tile_anns])
        
    return meta_tile_anns

def fetch_gcs_filenames(cache_gcs_path, output_file):
    p = subprocess.call(['gsutil', '-m', 'ls', '{:s}/*.svs/tiles/*.txt'.format(cache_gcs_path)], stdout=open(output_file, "w"))
    assert p == 0

    
def get_svs_metadata(cancertype,
                     all_svs_metadata = 'data/TCGA_slide_images_metadata.txt',
                     AppMags = [20, 40], sample_types = [1, 11], vendor = 'aperio'):
    svs_metadata = pd.read_csv(all_svs_metadata, index_col=0, low_memory=False)

    svs_metadata['CancerType'] = svs_metadata['GCSurl'].map(lambda x:x.split('/')[6][5:])
    svs_metadata['SampleID'] = svs_metadata['sample_barcode'].map(lambda s:int(s[13:15]))

    # only keep magnification 20,40x
    svs_metadata = svs_metadata[svs_metadata['AppMag'].isin(AppMags)]
    # only use primary solid tumor (1) 
    svs_metadata = svs_metadata[svs_metadata['SampleID'].isin(sample_types)] # solid primary tumor/solid normal
    # only use aperio images
    svs_metadata = svs_metadata[svs_metadata['vendor']==vendor]
    if not cancertype is None:
        svs_metadata = svs_metadata[svs_metadata['CancerType'] == cancertype.upper()]
    
    return svs_metadata
    
def remove_middle_qcut(image_files_metadata, drop_label=1):
    tmp = image_files_metadata.copy()
    task_class_counts_dict = {'mean_intensity': 3, 'std_intensity': 3}
    for label, class_count in task_class_counts_dict.items():
        tmp[label+'_label'] = pd.qcut(tmp[label], class_count,labels=False).astype(int)
        tmp = tmp[tmp[label+'_label'] != drop_label]        
        tmp[label+'_label'].replace(2, 1, inplace = True)    
    return tmp

def merge_cache_stats(cache_stats_directory):
    cache_stats = glob.glob(cache_stats_directory + '/*.txt')
    all_dfs = [pd.read_csv(s, index_col=0) for s in cache_stats]
    cache_stats = pd.concat(all_dfs)
    return cache_stats.reset_index()

def load_and_filter_slides_metadata(cancer_types=['LUAD'], optical_zoom=[40],
                                   sample_type_ids = [1, 11], vendor='aperio',
                                   digital_zoom = ('level_2__downsample',16),
                                   TCGA_metadata = DATA_PATH+'/TCGA_slide_images_metadata.txt'):
    
    df = pd.read_csv(TCGA_metadata, index_col=0, low_memory=False)
    df['CancerType'] = df['GCSurl'].map(lambda x:x.split('/')[6][5:])
    df = df[df['CancerType'].isin(cancer_types)]
    df['SampleID'] = df['sample_barcode'].map(lambda s:int(s[13:15]))

    df = df[df['AppMag'].isin(optical_zoom)]
    df = df[df['SampleID'].isin(sample_type_ids)]
    if vendor!=None:
        df = df[df['vendor']==vendor]
    if digital_zoom != None:
        df = df[df[digital_zoom[0]].fillna(0).astype(int)==digital_zoom[1]]
    # df = df.iloc[df.reset_index().groupby('case_barcode')['object_size'].idxmin()]
    df.sort_values('object_size', inplace=True,ascending=True)
    return df

def annotate_cache_df_by_coudray_snvs(cache_df, outputfile,
                                      CoudrayGenes = ['TP53','LRP1B','KRAS','KEAP1','FAT4','STK11','EGFR','FAT1','NF1','SETBP1']):
    print('Load gene information...')
    MutationCounts = get_per_sample_mutation_counts(maximum_number_of_output_genes=-1)
    print('Filter down excess genes...')
    MutationCounts = MutationCounts[CoudrayGenes]
    print('Select tumors...')
    image_files_metadata = cache_df[cache_df['is_tumor'] == 1]
    print('Merge snv data with tile data...')
    image_files_metadata_with_snvs = image_files_metadata.merge(
        (MutationCounts>0).astype(int).reset_index(), left_on='patient_id', right_on='CaseID', how='inner')
    print('Shuffle tiles...')
    image_files_metadata_with_snvs = image_files_metadata_with_snvs.sample(frac=1, random_state=0).reset_index(drop=True)
    print('Save results to disk...')
    image_files_metadata_with_snvs.to_csv(outputfile, index=False)
    print('Return output...')
    return image_files_metadata_with_snvs

def filter_maf(cancer_type, calculate_maf=False,
              filters_dict = {'IMPACT':['HIGH','MODERATE']},
              maf_files_glob_format='/home/jnh/TCGAdata/maf_files/*/TCGA.*.mutect.*.maf.gz'):
    '''
    this routine goes through `filters_dict` and searches if for each  
    key its value is in the list provided. The final result is
    logical AND of these checks, which determines, which lines
    of `maf` to keep. A sample `filters_dict` may look like:

        filters_dict = {'IMPACT':['HIGH','MODERATE'],
                        'Variant_Type': ['SNP'],
                        'Variant_Classification': ['Nonsense_Mutation'],
                        'Consequence': ['frameshift_variant','stop_gained','start_lost','stop_lost']}

    If more filters are needed on read depth and allele frequency, use the following:
    min_t_depth = 10
    min_n_depth = 10
    min_allele_freq = 0.1
    Filter = (maf['t_depth']>=min_t_depth) & (maf['n_depth']>=min_n_depth) 
    maf = maf[Filter]    
    maf['allele_freq'] = maf['t_alt_count']/maf['t_depth']
    maf = maf[maf['allele_freq'] >= min_allele_freq]
    '''
    
    maf_files = glob.glob(maf_files_glob_format)

    maffile = [s for s in maf_files if cancer_type in s][0]

    dtypes = {'Tumor_Sample_Barcode':str,'Hugo_Symbol':str,'Variant_Classification':str,
              'Variant_Type':str,'dbSNP_RS':str,'t_depth':int,'t_ref_count':int,'t_alt_count':int,
              'n_depth':int,'IMPACT':str,'COSMIC':str,'FILTER':str,'Consequence':str}

    maf = pd.read_csv(maffile,sep='\t',comment='#',dtype=dtypes,usecols=dtypes.keys())
    maf['CaseID'] = maf['Tumor_Sample_Barcode'].map(lambda x:x[:12])
    maf.drop('Tumor_Sample_Barcode',axis=1,inplace=True)
    maf_filter = True
    for key, value_list in filters_dict.items(): 
        maf_filter = maf_filter & (maf[key].isin(value_list))
    maf = maf[maf_filter]
    if calculate_maf:
        maf['maf'] = maf['t_alt_count']/(maf['t_alt_count'] + maf['t_ref_count'])
    return maf

def get_per_sample_mutation_counts(cancer_type, maximum_number_of_output_genes=20,
                                  filters_dict = {'IMPACT':['HIGH','MODERATE']},
                                  maf_files_glob_format='/home/jnh/TCGAdata/maf_files/*/TCGA.*.mutect.*.maf.gz'):
   
    maf = filter_maf(cancer_type, filters_dict = filters_dict, maf_files_glob_format=maf_files_glob_format)
 
    MutationCounts = maf.groupby(['CaseID','Hugo_Symbol']).size()
    MutationCounts = MutationCounts.unstack()
    
    if maximum_number_of_output_genes>0:
        SortedGenes = MutationCounts.notnull().sum().sort_values(ascending=False).index[:maximum_number_of_output_genes]
        MutationCounts = MutationCounts[SortedGenes]
    MutationCounts.fillna(0,inplace=True)
    return MutationCounts

def annotate_tile_metadata_by_snvs(tiles_basic_metadata, outputfile, gene_names, cancertype, maf_files_glob_format):
    print('Load gene information...')
    MutationCounts = get_per_sample_mutation_counts(cancertype, maximum_number_of_output_genes=-1, maf_files_glob_format=maf_files_glob_format)
    print('Filter down excess genes...')
    MutationCounts = MutationCounts[gene_names]
    print('Select tumors...')
    image_files_metadata = tiles_basic_metadata[tiles_basic_metadata['is_tumor'] == 1] # discard normal samples
    print('Merge snv data with tile data...')
    image_files_metadata_with_snvs = image_files_metadata.merge(
        (MutationCounts>0).astype(int).reset_index(), left_on='patient_id', right_on='CaseID', how='inner')
    print('Shuffle tiles...')
    image_files_metadata_with_snvs = image_files_metadata_with_snvs.sample(frac=1, random_state=0).reset_index(drop=True)
    print('Save results to disk...')
    image_files_metadata_with_snvs.to_csv(outputfile, index=False)
    print('Return output...')
    return image_files_metadata_with_snvs

def get_svs_metadata_from_bgq(project_id = 'jax-nihcc-res-00-0011', output_file = 'data/TCGA_slide_images_metadata.txt'):
    df_sizes = pd.read_gbq('SELECT * FROM [isb-cgc:metadata.gcs_listing_20171101] WHERE object_gcs_url LIKE "%.svs"', 
                           project_id=project_id)
    df = pd.read_gbq('SELECT * FROM [isb-cgc:metadata.TCGA_slide_images]', 
                     project_id=project_id)

    df = df.merge(df_sizes[['object_gcs_url','object_size']],left_on='file_gcs_url',right_on='object_gcs_url',how='left')
    df.set_index('case_barcode',inplace=True)
    df.drop('object_gcs_url',axis=1,inplace=True)
    df.to_csv(output_file)

def get_blob_val(blobname, bucket, apply_func = lambda x: x.tostring()):
    blob = bucket.blob(blobname)
    blob_val = blob.download_as_string()
    blob_val = np.fromstring(blob_val, dtype=np.float32, sep=',')
#     if convert_to_string:
#         blob_val = blob_val.tostring()
# #     blob_val = np.fromstring(blob_val, sep=',')
    return apply_func(blob_val)

def load_cache_values(image_files_metadata, notebook = True, tqdm_desc = '',
                      project='jax-nihcc-res-00-0011', bucket_name = None,
                      apply_func = lambda x: x.tostring(), user_project=None):
    assert bucket_name is not None, 'please provide bucket name'
    client = storage.Client(project=project)
    
    if user_project==None:
        user_project=project
    bucket = client.bucket(bucket_name, user_project=user_project)
    
    if notebook:
        tqdm_notebook.pandas(desc=tqdm_desc)
    else:
        tqdm.pandas(desc=tqdm_desc)
        
    cache_values = image_files_metadata['rel_path'].progress_apply(lambda x: get_blob_val(x, bucket, apply_func))
    return cache_values 
    
def merge_image_files_data_with_mutation_counts(MutationCounts, 
                                                image_file_metada_filename = 'data/image_files_metadata_tumor.txt'):
    raise('inspect this function')
    image_files_metadata = pd.read_csv(image_file_metada_filename,index_col=0)
    image_files_metadata = image_files_metadata.merge((MutationCounts>0).astype(int).reset_index(),on='sample_id',how='inner')
    return image_files_metadata

def calculate_intensity_stats(img):
    intensity_labels = {
        'mean_intensity':img.mean(),
        'std_intensity':img.std()
    }
    return pd.Series(intensity_labels)

def label_jpeg_files(task_class_counts_dict, validation_percentage, testing_percentage,
                    glob_path = 'tcga_tiles/luad_40x_level_2_16/TCGA-*.svs/tiles/tile_*.jpg',
                    cache_directory = 'tcga_tiles/luad_40x_level_2_16_cache',
                    BOTTLENECK_DATAFRAME_KEYWORD = 'rel_path',
                    IMAGE_DATAFRAME_KEYWORD = 'image_filename',
                    use_tqdm_notebook_widget = True, label_image_stats=True
                    ):
    
    image_filenames = glob.glob(glob_path)
    image_files_metadata = pd.DataFrame(image_filenames,columns=[IMAGE_DATAFRAME_KEYWORD])
    image_files_metadata['sample_id'] = image_files_metadata['image_filename'].map(lambda s:s.split('/')[-3].split('.')[0])
    image_files_metadata['is_tumor'] = image_files_metadata['sample_id'].apply(lambda s:int(s[13:15])<10).astype(int)
    
    strip_shared_path = image_files_metadata['image_filename'].str.split('/').apply(pd.Series)
    idx = np.where(strip_shared_path.apply(pd.Series.nunique)!=1)[0][0]
    image_files_metadata[BOTTLENECK_DATAFRAME_KEYWORD] = strip_shared_path.iloc[:,idx:].apply(lambda s:os.path.join(cache_directory,'/'.join(s)+'_cached.txt'),axis=1)
    image_files_metadata = image_files_metadata.sample(frac=1, random_state=0).reset_index(drop=True)
    assert validation_percentage + testing_percentage<100,"There are not enough training samples"
    getSHA1 = lambda s: hashlib.sha1(compat.as_bytes(s)).hexdigest()
    image_files_metadata['sample_id_SHA1'] = image_files_metadata['sample_id'].map(getSHA1)
    assert not image_files_metadata[['sample_id_SHA1','sample_id']].drop_duplicates()['sample_id_SHA1'].duplicated().any(),"SHA1 produced duplicates!!!"

    # The 'crossval_group' assigned to tiles may highly correlated training and testing data
    # To avoid that let's re-assign each patient to one category (training,testing,validation):
    MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1 # ~134M
    image_files_metadata['crossval_group'] = image_files_metadata['sample_id_SHA1'].apply(
        lambda x:(int(x, 16) % (MAX_NUM_IMAGES_PER_CLASS + 1))/ MAX_NUM_IMAGES_PER_CLASS*100)
    
    if validation_percentage > 0:
        image_files_metadata['crossval_group'] = pd.cut(image_files_metadata['crossval_group'],
               [-1, testing_percentage, testing_percentage + validation_percentage,100],
               labels=['testing','validation','training'])
    else:
        image_files_metadata['crossval_group'] = pd.cut(image_files_metadata['crossval_group'],
               [-1, testing_percentage, 100],
               labels=['testing','training'])

    if use_tqdm_notebook_widget:
        tqdm_notebook.pandas(desc='Labeling...')
    else:
        tqdm.pandas(desc='Labeling...')
    
    def read_tile_and_calculate_intensity_stats(x):
        try:
            x_array = plt.imread(x)
        except (OSError, TypeError):
            print('\nThe following file seems to be corrupted: {:s}\n'.format(x))
            x_array = np.array([np.nan])
        return calculate_intensity_stats(x_array)
    temp = image_files_metadata[IMAGE_DATAFRAME_KEYWORD].progress_apply(read_tile_and_calculate_intensity_stats)
    assert temp.shape[1]==len(task_class_counts_dict), "The number of tasks needs to match the number of fields produced by calculate_intensity_stats"
    image_files_metadata = pd.concat([image_files_metadata,temp],axis=1)
    for label,class_count in task_class_counts_dict.items():
        image_files_metadata[label+'_label'] = pd.qcut(image_files_metadata[label],class_count,labels=False)    
    return image_files_metadata
    
def label_cache_files(validation_percentage = 15, testing_percentage = 15, outputfile = 'data/cache_dataframe.txt',
                      task_class_counts_dict = {'mean_val':2, 'std_val':2},
                      glob_path = 'tcga_tiles/luad/filelist_luad_40x_level2_downsampl16_512x512_cache/*/tiles/tile*.jpg_cached.txt',
                      lstrip_string = 'gs://histology/', BOTTLENECK_DATAFRAME_KEYWORD = 'rel_path',
                      cache_gcs_paths = 'data/filelist_luad_40x_level2_downsampl16_512x512_cache.txt',
                      use_tqdm_notebook_widget = True, include_cache_stats=False, glob_locally = False):

    if use_tqdm_notebook_widget:
        tqdm_notebook.pandas(desc='')
    else:
        tqdm.pandas(desc='')

    if glob_locally:
        print('Globbing tile caches...')
        cache_df = glob.glob(glob_path)
        cache_df = pd.DataFrame(cache_df, columns=[BOTTLENECK_DATAFRAME_KEYWORD])
    else:
        print('Fetching cache filenames...')
        cache_df = pd.read_csv(cache_gcs_paths, sep=',', header = None)
        cache_df.columns = ['GCSurl']
        cache_df[BOTTLENECK_DATAFRAME_KEYWORD] = cache_df['GCSurl'].progress_apply(lambda x: x[len(lstrip_string):])
    
    print('Randomizing tiles...')
    cache_df = cache_df.sample(frac=1, random_state=0).reset_index(drop=True)

    print('Extracting sample ids...')
    cache_df['sample_id'] = cache_df[BOTTLENECK_DATAFRAME_KEYWORD].progress_apply(lambda s:s.split('/')[-3].split('.')[0])

    print('Extracting patient ids...')
    cache_df['patient_id'] = cache_df['sample_id'].progress_apply(lambda x: x[:12])
    
    print('Extracting tumor/normal label...')
    cache_df['is_tumor'] = cache_df['sample_id'].progress_apply(lambda s:int(s[13:15])<10).astype(int)

    assert validation_percentage + testing_percentage<100,"There are not enough training samples"

    getSHA1 = lambda s: hashlib.sha1(compat.as_bytes(s)).hexdigest()
    print('Hashing sample ids...')
    cache_df['sample_id_SHA1'] = cache_df['sample_id'].progress_apply(getSHA1)

    assert not cache_df[['sample_id_SHA1','sample_id']].drop_duplicates()['sample_id_SHA1'].duplicated().any(),"SHA1 produced duplicates!!!"

    # The 'crossval_group' assigned to tiles may highly correlated training and testing data
    # To avoid that let's re-assign each patient to one category (training,testing,validation):
    print('Assigning cross-validation labels...')
    MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1 # ~134M
    cache_df['crossval_group'] = cache_df['sample_id_SHA1'].progress_apply(
        lambda x:(int(x, 16) % (MAX_NUM_IMAGES_PER_CLASS + 1))/ MAX_NUM_IMAGES_PER_CLASS*100)

    if validation_percentage > 0:
        cache_df['crossval_group'] = pd.cut(cache_df['crossval_group'],
               [-1, testing_percentage, testing_percentage + validation_percentage,100],
               labels=['testing','validation','training'])
    else: # ignore validation set
        cache_df['crossval_group'] = pd.cut(cache_df['crossval_group'],
               [-1, testing_percentage, 100],
               labels=['testing','training'])

    if include_cache_stats:
        def get_cache_stats(cache_filename):
            x = np.loadtxt(cache_filename, delimiter=',')
            cache_stats = {
                'mean_val':x.mean(),
                'std_val':x.std()
            }
            return pd.Series(cache_stats)

        print('Calculate per cache statistics...')
        tmp = cache_df[BOTTLENECK_DATAFRAME_KEYWORD].progress_apply(get_cache_stats)

        print('Merging the results...')
        cache_df = pd.concat([cache_df, tmp],axis=1)

        assert tmp.shape[1]==len(task_class_counts_dict), "The number of tasks needs to match the number of fields produced by get_cache_stats"

        print('Creating cache stat labels...')
        for label, class_count in task_class_counts_dict.items():
            cache_df[label+'_label'] = pd.qcut(cache_df[label], class_count, labels=False)
    
    print('Saving cache dataframe to disk...')
    cache_df.to_csv(outputfile, index=False)
    print('Saved to: {:s}'.format(outputfile))
    return cache_df

def assign_validation_and_other_labels_to_tiles(training_percentage=70, testing_percentage=15, outputfile=None, glob_path=None,
                                                lstrip_string=None, task_class_counts_dict = {'mean_val':2, 'std_val':2},
                                                relative_path_keyword = 'rel_path', cache_gcs_paths=None, use_tqdm_notebook_widget = False,
                                                include_cache_stats=False, glob_locally = False,
                                                backward_count_to_samplename=3, drop_normals=False):
    '''
    Randomly assigns slides to train/test/validation, and creates a dataframe with tile paths
    these corresponding labels. Furthermore it adds some basic annotations such as patient id,
    a random hash, tumor/normal status, and some optional basic statistics (e.g. mean, std)
    of the image. A field with relative path is also construced from each GCSurl for downstream
    analysis.

    NOTE: calculating image statistics has been implemented for caches (tsv files), and needs to
    be implemented for JPEG files (see GH-57).

    Arguments:
    training_percentage (float): Percentage of samples in the training set
    testing_percentage (float): Percentage of samples in the test set
    outputfile (str): File to save the output dataframe. If None it will not save the output
    glob_path (str): glob path only used when locally reading the files. Works if glob_locally=True
        (default:None)
    lstrip_string: Prefix to be removed from GCSurl in order to create relative paths. In most cases
        it can be the bucket name.
    task_class_counts_dict (dict): Number of classes in the optional statistics. This is used
        to split the statistics into percentiles and label each with unique integers (0, 1, ...).
        Default: {'mean_val':2, 'std_val':2}
    relative_path_keyword (str): Column name used for the relative paths constructed from GCSurl
        (default: 'rel_path')
    cache_gcs_paths (str or pd.DataFrame): the list of GCSurls for all the tiles to be annotated. 
        If this is a text file where each row is a GCSurl then the argument would be path of to that text file. 
        Alternatively it can be a dataframe with a similar structure.
    use_tqdm_notebook_widget (bool): flag to make tqdm work with notebooks (default: False)
    include_cache_stats (bool): set this to True in order to calculate tile statistics (e.g. image mean,
        std). Note that this process can be quite time-consuming (default: False).
    glob_locally (bool): The function is able to construct the annotations from local folder structure
        instead of GCS (default: False)

    Returns:
    cache_df (pandas.DataFrame): Dataframe containing GCSurls and their annotations.
    '''

    validation_percentage = 100 - testing_percentage - training_percentage
    # GH-79
  
    if use_tqdm_notebook_widget:
        tqdm_notebook.pandas(desc='')
    else:
        tqdm.pandas(desc='')

    if glob_locally:
        print('Globbing tiles locally...')
        cache_df = glob.glob(glob_path)
        cache_df = pd.DataFrame(cache_df, columns=[relative_path_keyword])
    else:
        cache_df = util.read_csv(cache_gcs_paths, columns=['GCSurl'], sep=',', header = None)
        cache_df[relative_path_keyword] = cache_df['GCSurl'].progress_apply(lambda x: x[len(lstrip_string):])

    print('Randomizing tiles...')
    cache_df = cache_df.sample(frac=1, random_state=0).reset_index(drop=True)

    print('Extracting sample ids...')
    cache_df['sample_id'] = cache_df[relative_path_keyword].progress_apply(lambda s:s.split('/')[-backward_count_to_samplename].split('.')[0])

    print('Extracting patient ids...')
    cache_df['patient_id'] = cache_df['sample_id'].progress_apply(lambda x: x[:12])

    print('Extracting tumor/normal label...')
    cache_df['is_tumor'] = cache_df['sample_id'].progress_apply(lambda s:int(s[13:15])<10).astype(int)

    assert validation_percentage + testing_percentage<100,"There are not enough training samples"

    if drop_normals:
        print('Dropping normal samples from the list...')
        cache_df = cache_df[cache_df['is_tumor']==1]

    print('Extracting slide preparation method...')
    cache_df['slide_code'] = cache_df['sample_id'].map(lambda x: x.split('-')[-1])
    cache_df['tissue-method'] = cache_df['slide_code'].map(lambda x: ['Frozen', 'FFPE'][x.startswith('DX')])
        
    getSHA1 = lambda s: hashlib.sha1(compat.as_bytes(s)).hexdigest()
    print('Hashing sample ids...')
    cache_df['sample_id_SHA1'] = cache_df['sample_id'].progress_apply(getSHA1)

    assert not cache_df[['sample_id_SHA1','sample_id']].drop_duplicates()['sample_id_SHA1'].duplicated().any(),"SHA1 produced duplicates!!!"

    # The 'crossval_group' assigned to tiles may highly correlated training and testing data
    # To avoid that let's re-assign each patient to one category (training,testing,validation):
    print('Assigning cross-validation labels to samples...')
    MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1 # ~134M, need a huge number.
    cache_df['crossval_group'] = cache_df['sample_id_SHA1'].progress_apply(
        lambda x:(int(x, 16) % (MAX_NUM_IMAGES_PER_CLASS + 1))/ MAX_NUM_IMAGES_PER_CLASS*100)

    if validation_percentage > 0:
        cache_df['crossval_group'] = pd.cut(cache_df['crossval_group'],
               [-1, testing_percentage, testing_percentage + validation_percentage,100],
               labels=['testing','validation','training'])
    else: # ignore validation set
        cache_df['crossval_group'] = pd.cut(cache_df['crossval_group'],
               [-1, testing_percentage, 100],
               labels=['testing','training'])

    if include_cache_stats:
        def get_cache_stats(cache_filename):
            x = np.loadtxt(cache_filename, delimiter=',')
            cache_stats = {
                'mean_val':x.mean(),
                'std_val':x.std()
            }
            return pd.Series(cache_stats)

        print('Calculate per tile statistics...')
        tmp = cache_df[relative_path_keyword].progress_apply(get_cache_stats)

        print('Merging the results...')
        cache_df = pd.concat([cache_df, tmp],axis=1)

        assert tmp.shape[1]==len(task_class_counts_dict), "The number of tasks needs to match the number of fields produced by get_cache_stats"

        print('Creating tile stat labels...')
        for label, class_count in task_class_counts_dict.items():
            cache_df[label+'_label'] = pd.qcut(cache_df[label], class_count, labels=False)

    if outputfile is not None:
        print('Saving tile dataframe to disk...')
        cache_df.to_csv(outputfile, index=False)
        print('Saved to: {:s}'.format(outputfile))
    return cache_df

def get_holdout_splits(df, training_percentage, testing_percentage):
    return pd.cut(df['crossval_group'],
               [-1, training_percentage, training_percentage + testing_percentage, 101],
               labels=['training','testing','validation'])

def stratified_crossval(image_files_metadata, training_percentage=70, testing_percentage=15, label_name='label'):
    image_files_metadata['crossval_group'] = image_files_metadata['sample_id_SHA1'].apply(lambda x:int(x, 16))
    image_files_metadata['crossval_group'] = image_files_metadata.groupby(label_name)['crossval_group'].apply(lambda x: (x-x.min())/(x.max()-x.min())*100)

    image_files_metadata['crossval_group'] = image_files_metadata.groupby(label_name, sort=False
                                                                         ).apply(lambda x: get_holdout_splits(x, training_percentage, testing_percentage)
                                                                                ).reset_index(level=0, drop=True)
    
    return image_files_metadata
