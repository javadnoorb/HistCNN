from google.cloud import storage
import glob
import os
from histcnn import util

def parse_gcs_path(gcs_path):
    assert 'gs://' in gcs_path, 'GCS paths must start with gs://'
    fields = gcs_path.split('/')
    output = {}
    output['bucket'] = fields[2]
    output['path'] = '/'.join(fields[3:])
    return output

def get_project_id():
    cmd = "gcloud config list --format value(core.project)"
    project_id =  util.run_unix_cmd(cmd)[0] 
    return project_id

class gcsbucket:
    def __init__(self, project_name=None, bucket_name=None):
        if project_name is None:
#           project_name = get_project_id()
            raise Exception('Please provide the project id.')
        self.project_name = project_name
        self.bucket_name = bucket_name
        self.make_bucket()
        
    def make_bucket(self):    
        client = storage.Client(project = self.project_name)
        self.bucket = client.get_bucket(self.bucket_name)        
        
    def _copy_single_file_to_gcs(self, input_file, output_file, verbose=False):
        if verbose:
            print('Copying %s ====> %s'%(input_file, output_file))
        blob = self.bucket.blob(output_file)
        blob.upload_from_filename(input_file)

    def _copy_glob_files_to_gcs(self, input_globfiles, output_path, recursive = True, verbose=False):
        for input_file in glob.glob(input_globfiles):
            output_file = os.path.join(output_path, os.path.basename(input_file))
            if os.path.isdir(input_file):
                if recursive:
                    self._copy_glob_files_to_gcs(os.path.join(input_file,'*'), output_file+'/', recursive=recursive, verbose=verbose)
                else:
                    continue
            else:
                self._copy_single_file_to_gcs(input_file, output_file, verbose=verbose)
    
    def copy_files_to_gcs(self, input_path, output_path, recursive=True, verbose=False):
        assert 'gs://' not in output_path, "Provide relative bucket path."
        if os.path.isdir(input_path):
            self._copy_glob_files_to_gcs(input_path+'/*', output_path, recursive=recursive, verbose=verbose)
        elif '*' in input_path:
            self._copy_glob_files_to_gcs(input_path, output_path, recursive=recursive, verbose=verbose)
        else:
            self._copy_single_file_to_gcs(input_path, output_path, verbose=verbose)
    
    def download_from_gcs(self, gcs_file_path, output_dir='.'):
        assert 'gs://' not in gcs_file_path, "Provide relative bucket path."
        filename = gcs_file_path.split('/')[-1]
        blob = self.bucket.blob(gcs_file_path)
        destination_full_file_name = os.path.join(util.mkdir_if_not_exist(output_dir), filename)
        blob.download_to_filename(destination_full_file_name)

    def list_blob(self, prefix):
        blobs = self.bucket.list_blobs(prefix=prefix)
        return list(blobs)

    def is_available(self, prefix):
        blobs = self.list_blob(prefix)
        return len(blobs) == 1
    

