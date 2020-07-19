import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path


def get_data(input_path):
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

    i = 0
    with open(input_path, 'r') as f:
        print('Parsing annotation files')

        for line in f:
            # i += 1
            # if i == 60:
            #     break
            line_split = line.strip().split(',')
            (filename, class_name, x1, y1, x2, y2) = line_split

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print(
                        'Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}

                img = cv2.imread(filename)
                (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
                if np.random.randint(0, 6) > 0:
                    all_imgs[filename]['imageset'] = 'train'
                else:
                    all_imgs[filename]['imageset'] = 'val'

            all_imgs[filename]['bboxes'].append(
                {'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping


def read_data_from_xml(input_path='./data/TEXT_ANNOTATED/'):
    file_path = Path(input_path)
    i = 0

    with open('./new_data/text_annotated.txt', 'w') as f:
        for path in file_path.glob('*.xml'):
            root = ET.parse(str(path)).getroot()
            file_name = root.find('filename').text
            img = cv2.imread(input_path + file_name)
            re_name = './new_data/img_' + str(i) + '.png'
            i = i + 1
            cv2.imwrite(re_name, img)
            for title in root.iter('object'):
                # title_name = title.find('name').text
                bndbox = title.find('bndbox')
                xmin = bndbox.find('xmin').text
                ymin = bndbox.find('ymin').text
                xmax = bndbox.find('xmax').text
                ymax = bndbox.find('ymax').text
                f.write(re_name + ',' + 'text' + ',' + xmin + ',' + ymin + ',' + xmax + ',' + ymax + '\n')

