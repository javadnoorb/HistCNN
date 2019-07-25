
from histcnn import handle_google_cloud_apis
import sys

def checkpath(args):
    blobpath_dict = handle_google_cloud_apis.parse_gcs_path(args.checkpath)
    gcsbucket = handle_google_cloud_apis.gcsbucket(project_name=args.project,
                                                bucket_name=blobpath_dict['bucket'])
    file_available = gcsbucket.is_available(blobpath_dict['path'])
    if file_available and args.download_dir is not None:
        gcsbucket.download_from_gcs(blobpath_dict['path'], output_dir=args.download_dir)

    output_status=1-file_available
    if args.print_status:
        print(output_status)
    sys.exit(output_status)


