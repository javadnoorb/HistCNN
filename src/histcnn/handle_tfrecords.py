import tensorflow as tf
import tqdm 
import numpy as np
from PIL import Image

def create_tfrecords_per_category_for_caches(image_files_metadata, label_names, category, notebook = False,
                                  tfrecord_file_name_prefix = 'image_files_metadata.tfrecord'):
    assert category in image_files_metadata['crossval_group'].values, "Unrecognized category provided."
 
    create_tfrecords_from_caches(image_files_metadata[image_files_metadata['crossval_group'] == category],
                     label_names,                     
                     "{:s}.{:s}".format(tfrecord_file_name_prefix, category), notebook=notebook)

def create_tfrecords_per_category_for_tiles(image_files_metadata, label_names, category,
                                  tfrecord_file_name_prefix = 'image_files_metadata.tfrecord'):
    assert category in image_files_metadata['crossval_group'].values, "Unrecognized category provided."
 
    create_tfrecords_from_tiles(image_files_metadata[image_files_metadata['crossval_group'] == category],
                                label_names,
                                "{:s}.{:s}".format(tfrecord_file_name_prefix, category))

def create_tfrecords_from_caches(image_files_metadata, label_names,
                     tfrecordsfile, notebook = True, get_cache_values_from_file = False):
    def _int64_feature(value):
          return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(value):
          return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _bytes_feature(value):
          return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    if notebook:
        tqdmrangefunc = tqdm.tnrange
    else:
        tqdmrangefunc = tqdm.trange
    
    include_image_filename = 'imagefilename' in image_files_metadata.columns  
    
    with tf.python_io.TFRecordWriter(tfrecordsfile) as writer:
        for i in tqdmrangefunc(len(image_files_metadata)):
            # Load the image
            sampleid = image_files_metadata['sample_id'].iloc[i]
            if include_image_filename:
                imagefilename = image_files_metadata['image_filename'].iloc[i]
            else:
                imagefilename = ''
                
            cachefile = image_files_metadata['rel_path'].iloc[i]
            
            if get_cache_values_from_file:
                with open(cachefile,'r') as f:
                    cache_values = f.readline()
                cache_values = np.fromstring(cache_values, dtype=np.float32, sep=',').tostring()
            else:
                cache_values = image_files_metadata['cache_values'].iloc[i]
                
            feature = {'sampleid':_bytes_feature(sampleid.encode('utf-8')),
                       'imagefilename':_bytes_feature(imagefilename.encode('utf-8')),
                       'cachefile':_bytes_feature(cachefile.encode('utf-8')),
                       'image': _bytes_feature(cache_values)}

            label_list = image_files_metadata[label_names].iloc[i].values
            label_features = {label_name: _int64_feature(label)
                              for label, label_name in zip(label_list, label_names)}
                  
            feature.update(label_features)
            # Create an example protocol buffer
            example = tf.train.Example(features = tf.train.Features(feature=feature))
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())


def create_tfrecords_from_tiles(image_files_metadata, label_names, tfrecordsfile, add_extra_zero=True):
    '''
    Creates tfrecord files from a list of annotated image tiles
    '''

    def list_if_scalar(value):
        if isinstance(value, list):
            return value
        else:
            return [value]

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list_if_scalar(value)))
    def _float_feature(value):
          return tf.train.Feature(float_list=tf.train.FloatList(value=list_if_scalar(value)))
    def _bytes_feature(value):
          return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_if_scalar(value)))

    tqdmrangefunc = tqdm.trange
    with tf.python_io.TFRecordWriter(tfrecordsfile) as writer:
        for i in tqdmrangefunc(len(image_files_metadata)):
            # Load the image
            sampleid = image_files_metadata['sample_id'].iloc[i]
            imagefilename = image_files_metadata['rel_path'].iloc[i]

            jpgfile = Image.open(imagefilename)
            image_buffer = tf.read_file(imagefilename)
            with tf.gfile.FastGFile(imagefilename, 'rb') as f:
                image_buffer = f.read()
            label_values = image_files_metadata[label_names].iloc[i].tolist()
            if add_extra_zero:
                label_values = [0]+label_values
            label_texts = [s.encode() for s in label_names]
        
            feature = {'metadata/sampleid':_bytes_feature(sampleid.encode('utf-8')),
                       'image/filename':_bytes_feature(imagefilename.encode('utf-8')),
                       'image/width': _int64_feature(jpgfile.width),
                       'image/height': _int64_feature(jpgfile.width),
                       'image/class/label': _int64_feature(label_values),
                       'image/class/text': _bytes_feature(label_texts),
                       'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}

            example = tf.train.Example(features = tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())       
