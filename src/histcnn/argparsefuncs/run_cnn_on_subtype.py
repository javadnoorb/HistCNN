import glob
import os
import sys
import pandas as pd
import numpy as np
import pickle

from histcnn import (util,
                 run_classification,
                 handle_tfrecords)

def run_subtype_classification(args):
    image_file_metadata_filename = os.path.join('/sdata', args.annotations_path, 'subtype_ann_{:s}_tumor.txt'.format(args.tissue))
    output_folder = os.path.join('/sdata', args.results_path,'{:s}/'.format(args.tissue))
    trecords_prefix = os.path.join('/sdata', args.pancancer_tfrecords_path, '{:s}/tfrecord'.format(args.tissue))
    saved_model_path = output_folder + 'saved_models/mychckpt'
    tensorboard_path = output_folder + 'tensorboard_logs/'
    train_test_percentage = args.train_test_percentage
    if args.treat_validation_as_test:
        train_test_percentage[1] = 100-args.train_test_percentage[0]
    pickle_path = output_folder + 'pickles/pickles_train{:d}_test{:d}/run_cnn_output.pkl'.format(*train_test_percentage)


    label_names = ['label']
    tfrec_glob_path = '{:s}*'.format(trecords_prefix)
    tfrecordfiles = glob.glob(tfrec_glob_path)
    assert len(tfrecordfiles)>0, tfrec_glob_path
    num_tfrecords = int(len(tfrecordfiles)/3)

    tfrecordfiles_dict = {s: ['{:s}{:d}.{:s}'.format(trecords_prefix, n, s) for n in range(num_tfrecords)] for s in ['training', 'testing', 'validation']}
    image_files_metadata = pd.read_csv(image_file_metadata_filename, sep=',')
    
    nClass = image_files_metadata['label'].max()+1

    if args.treat_validation_as_test:
        image_files_metadata['crossval_group'].replace('validation', 'testing', inplace=True)
        tfrecordfiles_dict['testing'] = tfrecordfiles_dict['testing']+tfrecordfiles_dict.pop('validation')

    test_batch_size = (image_files_metadata['crossval_group'] == 'testing').sum()
    if args.is_weighted:
        label_ratio = image_files_metadata[label_names].mean()
        pos_weight = (1/label_ratio - 1).tolist()
    else:
        pos_weight = 1

    class_probs = image_files_metadata.loc[image_files_metadata['crossval_group'] == 'training', label_names[0]].value_counts(normalize=True, sort=False).sort_index().values

    test_accuracies_list, predictions_list, confusion_matrices_list, imagefilenames, final_softmax_outputs_list = \
    run_classification.run_multilabel_classification_with_inception_CNN(label_names, tfrecordfiles_dict,
                                                                        test_batch_size=test_batch_size, nClass=nClass,
                                                                        train_batch_size = 512,
                                                                        how_many_training_steps=args.training_steps,
                                                                        avoid_gpu_for_testing=args.avoid_gpu_for_testing,
                                                                        do_not_train = args.do_not_train, pos_weight =
                                                                        pos_weight, dropout_keep_prob = args.dropout_keep_prob,
                                                                        saved_model_path = saved_model_path,
                                                                        summaries_dir = tensorboard_path, optimizer = args.optimizer, 
                                                                        class_probs=class_probs)

    util.mkdir_if_not_exist(os.path.dirname(pickle_path))

    pickle.dump([image_files_metadata, test_accuracies_list, predictions_list, confusion_matrices_list, imagefilenames, final_softmax_outputs_list],
		open(pickle_path, 'wb'))

