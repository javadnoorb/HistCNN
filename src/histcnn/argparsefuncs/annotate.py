from histcnn import choose_input_list
import pandas as pd
import pkg_resources

DATA_PATH = pkg_resources.resource_filename('histcnn', 'data/')

def annotate_inputs(args):
    if args.tissue is not None:
        gcs_path_list = choose_input_list.get_file_list_for_subtypes(args.tissue,
        caches_gcs_path_list_path = args.gcs_path_list,
        histotypes_counts_file = DATA_PATH+'/histotypes_counts_annotated.xlsx',
        sheet_number=2, verbose=True)
        output_file = None
    else:
        gcs_path_list = args.gcs_path_list
        output_file = args.output_file

    cache_df = choose_input_list.assign_validation_and_other_labels_to_tiles(training_percentage = args.train_test_percentage[0],
                                                  testing_percentage = args.train_test_percentage[1],
                                                  outputfile = output_file,
                                                  cache_gcs_paths = gcs_path_list, drop_normals = args.drop_normals,
                                                  lstrip_string = args.lstrip_string, backward_count_to_samplename=2)

    if args.tissue is not None:
        output_file = args.output_file
        cache_df = choose_input_list.annotate_subtypes(cache_df, args.tissue,
                        filename = DATA_PATH+'/histotypes.txt')
        print('Saving tile dataframe to disk...')
        cache_df.to_csv(output_file, index=False)
        print('Saved to: {:s}'.format(output_file))
    
    if args.her2_amplification:
        print('Annotating...')
        her2cnas = choose_input_list.get_her2_metadata()
        cache_df = cache_df.merge(her2cnas, on='patient_id')
        cache_df = cache_df.sample(frac=1, random_state=0).reset_index(drop=True)
        cache_df.to_csv(output_file, index=False)
        print('Saved to: {:s}'.format(output_file))


