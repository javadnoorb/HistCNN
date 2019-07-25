import openslide
import os
from histcnn import process_files

class hist_slide:
    def __init__(self, slide_info, tmp_path='tmp/', tile_size=512):
        self.tmp_path = tmp_path
        self.makedir(self.tmp_path)
        self.svsFile = slide_info['svsFilename']
        self.svsFullPath = os.path.join(self.tmp_path, self.svsFile)
        print('downloading svs file')
        process_files.DownloadSVS(slide_info['GCSurl'], outdir=tmp_path)
        self.Slide = openslide.OpenSlide(self.svsFullPath)
        self.tiles = openslide.deepzoom.DeepZoomGenerator(self.Slide, tile_size = tile_size, overlap=0, limit_bounds=False)
        self.AppMag = slide_info['AppMag']

    def clean(self):
        print('deleting svs file')
        os.remove(self.svsFullPath)

    def get_thumbnail(self, thumbsize = (400, 400)):
        return self.Slide.get_thumbnail(thumbsize)

    def get_tile(self, tile_level_step_down, tile_address):
        tile_level = self.tiles.level_count - tile_level_step_down
        tile = self.tiles.get_tile(tile_level, tile_address)
        return tile

    def makedir(self, directory):
        if not os.path.isdir(directory):
            os.makedirs(directory)
