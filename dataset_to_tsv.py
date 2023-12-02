"""
Adapt the prepare_coco_test() function to work for the VizWiz dataset.
The format of the TSV files is:
image_tsv: image-id, image as base64 encoded string
question_tsv: image-id, question text (lowercase)
Author: Jan Willruth
"""

import base64
import os.path as op
import json
from generativeimage2text.common import json_dump, read_to_buffer
from generativeimage2text.tsv_io import tsv_writer


def main():
    print('Converting dataset to TSV files...')
    subset = 'test'
    image_folder = f'dataset/val2014'
    json_file = f'dataset/annotations/captions_val2014.json'
    infos = json.loads(read_to_buffer(json_file))

    def gen_img_rows():
        for idx, inf in enumerate(infos['images']):
            idx = str(idx).zfill(4)
            payload = base64.b64encode(read_to_buffer(op.join(image_folder, inf['file_name'])))
            yield idx, payload
    
    tsv_writer(gen_img_rows(), f'dataset/coco_val2014.tsv')

    def gen_question_rows():
        for idx, inf in enumerate(infos['annotations']):
            idx = str(idx).zfill(4)
            caption = [{'caption_id': idx, 'caption': inf['caption'].lower()}]
            yield idx, json_dump(caption)
    tsv_writer(gen_question_rows(), f'dataset/caption_coco_val2014.tsv')

    return 'Finished creating tsv files!'


if __name__ == '__main__':
    print(main())
    print('All done!')