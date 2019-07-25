import glob
import pandas as pd
import os
import numpy as np
from tqdm import tqdm 

def get_pancancer_histotypes(input_path = '/sdata/data/TCGAdata/clinical/subtypes/', 
                             outputfile = '/sdata/data/pancancer_annotations/histotypes.txt'):
    clinical_files = glob.glob(os.path.join(input_path, '*/nationwidechildrens.org_clinical_patient*.txt'))
    clinical_files = pd.DataFrame(clinical_files, columns=['filename'])
    clinical_files.index = clinical_files['filename'].map(lambda x: x.split('_')[-1][:-4].upper())
    clinical_files.index.name = 'cancertype'
    histological_type = 'CDE_ID:3081934'
    clinical_files[histological_type] = True
    clinical_files['histological_type_names']  = np.nan
    for cancertype in clinical_files.index:
        file = clinical_files.loc[cancertype, 'filename']
        df = pd.read_csv(file, sep='\t', nrows=0, header=[0, 1, 2])
        has_hist_col = (histological_type in df.columns.get_level_values(2))
        clinical_files.loc[cancertype, histological_type] = has_hist_col
        if has_hist_col:
            histological_type_col = np.where(df.columns.get_level_values(2) == histological_type)[0][0]
    #         print(cancertype, histological_type_col)
            clinical_files.loc[cancertype, 'histological_type_col'] = histological_type_col
            clinical_files.loc[cancertype, 'histological_type_names'] = ', '.join(df.columns[histological_type_col][:2])
            assert df.columns[histological_type_col][1] == 'histological_type', 'improper column name'
    #         print(df.columns[histological_type_col])
    print('The following cancers do not have subtype column and will be removed from the analysis:')
    print(', '.join(clinical_files.loc[~clinical_files['CDE_ID:3081934']].index))

    clinical_files = clinical_files.loc[clinical_files['CDE_ID:3081934'], 'filename']

    histotypes = pd.DataFrame([])

    for cancertype in tqdm(clinical_files.index):
        file = clinical_files[cancertype]
        df = pd.read_csv(file, sep='\t', skiprows=[0, 2], header=0, na_values=['[Not Available]'],
                         usecols=['bcr_patient_barcode', 'histological_type'])
        df['cancertype'] = cancertype
        histotypes = pd.concat([histotypes, df])

    histotypes.set_index('bcr_patient_barcode', inplace=True)
    all_cancers = histotypes['cancertype'].unique()
    histotypes.dropna(inplace=True)

    print('The following cancers do not have informative subtype annotations and will be removed from the analysis:')
    print(', '.join(set(all_cancers) - set(histotypes['cancertype'])))

    histotypes['cancertype'] = histotypes['cancertype'].str.lower()
    histotypes.to_csv(outputfile, sep='\t')
    return histotypes
