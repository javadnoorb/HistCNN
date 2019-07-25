#!/usr/bin/env python
import pandas as pd
from histcnn import choose_input_list
import os
import sys

bucket_name = os.environ['caches_input_bucket']
cancertype = os.environ['cancertype']
validation_percentage = int(os.environ['validation_percentage'])
testing_percentage = int(os.environ['testing_percentage'])
lstrip_string = 'gs://' + bucket_name + '/'
temp_path = sys.argv[1]

cache_df = choose_input_list.assign_validation_and_other_labels_to_tiles(validation_percentage = validation_percentage,
                                                testing_percentage = testing_percentage,
                                                outputfile = temp_path+'/caches_basic_annotations.txt',
					        cache_gcs_paths = temp_path+'/caches_gcs_path_list.txt',
                                                lstrip_string = lstrip_string, backward_count_to_samplename=2)
