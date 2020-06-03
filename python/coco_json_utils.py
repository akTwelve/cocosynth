#!/usr/bin/python

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from skimage import measure, io
from shapely.geometry import Polygon, MultiPolygon
from PIL import Image

class InfoJsonUtils():
    """ Creates an info object to describe a COCO dataset
    """
    def create_coco_info(self, description, url, version, year, contributor, date_created):
        """ Creates the "info" portion of COCO json
        """
        info = dict()
        info['description'] = description
        info['url'] = url
        info['version'] = version
        info['year'] = year
        info['contributor'] = contributor
        info['date_created'] = date_created

        return info

class LicenseJsonUtils():
    """ Creates a license object to describe a COCO dataset
    """
    def create_coco_license(self, url, license_id, name):
        """ Creates the "licenses" portion of COCO json
        """
        lic = dict()
        lic['url'] = url
        lic['id'] = license_id
        lic['name'] = name

        return lic

class CategoryJsonUtils():
    """ Creates a category object to describe a COCO dataset
    """
    def create_coco_category(self, supercategory, category_id, name):
        category = dict()
        category['supercategory'] = supercategory
        category['id'] = category_id
        category['name'] = name

        return category

class ImageJsonUtils():
    """ Creates an image object to describe a COCO dataset
    """
    def create_coco_image(self, image_path, image_id, image_license):
        """ Creates the "image" portion of COCO json
        """
        # Open the image and get the size
        image_file = Image.open(image_path)
        width, height = image_file.size

        image = dict()
        image['license'] = image_license
        image['file_name'] = image_path.name
        image['width'] = width
        image['height'] = height
        image['id'] = image_id

        return image

class AnnotationJsonUtils():
    """ Creates an annotation object to describe a COCO dataset
    """
    def __init__(self):
        self.annotation_id_index = 0

    def create_coco_annotations(self, image_mask_path, image_id, category_ids):
        """ Takes a pixel-based RGB image mask and creates COCO annotations.
        Args:
            image_mask_path: a pathlib.Path to the image mask
            image_id: the integer image id
            category_ids: a dictionary of integer category ids keyed by RGB color (a tuple converted to a string)
                e.g. {'(255, 0, 0)': {'category': 'owl', 'super_category': 'bird'} }
        Returns:
            annotations: a list of COCO annotation dictionaries that can
            be converted to json. e.g.:
            {
                "segmentation": [[101.79,307.32,69.75,281.11,...,100.05,309.66]],
                "area": 51241.3617,
                "iscrowd": 0,
                "image_id": 284725,
                "bbox": [68.01,134.89,433.41,174.77],
                "category_id": 6,
                "id": 165690
            }
        """
        # Set class variables
        self.image_id = image_id
        self.category_ids = category_ids

        # Make sure keys in category_ids are strings
        for key in self.category_ids.keys():
            if type(key) is not str:
                raise TypeError('category_ids keys must be strings (e.g. "(0, 0, 255)")')
            break

        # Open and process image
        self.mask_image = Image.open(image_mask_path)
        self.mask_image = self.mask_image.convert('RGB')
        self.width, self.height = self.mask_image.size

        # Split up the multi-colored masks into multiple 0/1 bit masks
        self._isolate_masks()

        # Create annotations from the masks
        self._create_annotations()

        return self.annotations

    def _isolate_masks(self):
        # Breaks mask up into isolated masks based on color

        self.isolated_masks = dict()
        for x in range(self.width):
            for y in range(self.height):
                pixel_rgb = self.mask_image.getpixel((x,y))
                pixel_rgb_str = str(pixel_rgb)

                # If the pixel is any color other than black, add it to a respective isolated image mask
                if not pixel_rgb == (0, 0, 0):
                    if self.isolated_masks.get(pixel_rgb_str) is None:
                        # Isolated mask doesn't have its own image yet, create one
                        # with 1-bit pixels, default black. Make room for 1 pixel of
                        # padding on each edge to allow the contours algorithm to work
                        # when shapes bleed up to the edge
                        self.isolated_masks[pixel_rgb_str] = Image.new('1', (self.width + 2, self.height + 2))

                    # Add the pixel to the mask image, shifting by 1 pixel to account for padding
                    self.isolated_masks[pixel_rgb_str].putpixel((x + 1, y + 1), 1)

    def _create_annotations(self):
        # Creates annotations for each isolated mask

        # Each image may have multiple annotations, so create an array
        self.annotations = []
        for key, mask in self.isolated_masks.items():
            annotation = dict()
            annotation['segmentation'] = []
            annotation['iscrowd'] = 0
            annotation['image_id'] = self.image_id
            if not self.category_ids.get(key):
                print(f'category color not found: {key}; check for missing category or antialiasing')
                continue
            annotation['category_id'] = self.category_ids[key]
            annotation['id'] = self._next_annotation_id()

            # Find contours in the isolated mask
            mask = np.asarray(mask, dtype=np.float32)
            contours = measure.find_contours(mask, 0.5, positive_orientation='low')

            polygons = []
            for contour in contours:
                # Flip from (row, col) representation to (x, y)
                # and subtract the padding pixel
                for i in range(len(contour)):
                    row, col = contour[i]
                    contour[i] = (col - 1, row - 1)

                # Make a polygon and simplify it
                poly = Polygon(contour)
                poly = poly.simplify(1.0, preserve_topology=False)

                if (poly.area > 16): # Ignore tiny polygons
                    if (poly.geom_type == 'MultiPolygon'):
                        # if MultiPolygon, take the smallest convex Polygon containing all the points in the object
                        poly = poly.convex_hull

                    if (poly.geom_type == 'Polygon'): # Ignore if still not a Polygon (could be a line or point)
                        polygons.append(poly)
                        segmentation = np.array(poly.exterior.coords).ravel().tolist()
                        annotation['segmentation'].append(segmentation)

            if len(polygons) == 0:
                # This item doesn't have any visible polygons, ignore it
                # (This can happen if a randomly placed foreground is covered up
                #  by other foregrounds)
                continue

            # Combine the polygons to calculate the bounding box and area
            multi_poly = MultiPolygon(polygons)
            x, y, max_x, max_y = multi_poly.bounds
            self.width = max_x - x
            self.height = max_y - y
            annotation['bbox'] = (x, y, self.width, self.height)
            annotation['area'] = multi_poly.area

            # Finally, add this annotation to the list
            self.annotations.append(annotation)

    def _next_annotation_id(self):
        # Gets the next annotation id
        # Note: This is not a unique id. It simply starts at 0 and increments each time it is called

        a_id = self.annotation_id_index
        self.annotation_id_index += 1
        return a_id

class CocoJsonCreator():
    def validate_and_process_args(self, args):
        """ Validates the arguments coming in from the command line and performs
            initial processing
        Args:
            args: ArgumentParser arguments
        """
        # Validate the mask definition file exists
        mask_definition_file = Path(args.mask_definition)
        if not (mask_definition_file.exists and mask_definition_file.is_file()):
            raise FileNotFoundError(f'mask definition file was not found: {mask_definition_file}')

        # Load the mask definition json
        with open(mask_definition_file) as json_file:
            self.mask_definitions = json.load(json_file)

        self.dataset_dir = mask_definition_file.parent

        # Validate the dataset info file exists
        dataset_info_file = Path(args.dataset_info)
        if not (dataset_info_file.exists() and dataset_info_file.is_file()):
            raise FileNotFoundError(f'dataset info file was not found: {dataset_info_file}')

        # Load the dataset info json
        with open(dataset_info_file) as json_file:
            self.dataset_info = json.load(json_file)

        assert 'info' in self.dataset_info, 'dataset_info JSON was missing "info"'
        assert 'license' in self.dataset_info, 'dataset_info JSON was missing "license"'

    def create_info(self):
        """ Creates the "info" piece of the COCO json
        """
        info_json = self.dataset_info['info']
        iju = InfoJsonUtils()
        return iju.create_coco_info(
            description = info_json['description'],
            version = info_json['version'],
            url = info_json['url'],
            year = info_json['year'],
            contributor = info_json['contributor'],
            date_created = info_json['date_created']
        )

    def create_licenses(self):
        """ Creates the "license" portion of the COCO json
        """
        license_json = self.dataset_info['license']
        lju = LicenseJsonUtils()
        lic = lju.create_coco_license(
            url = license_json['url'],
            license_id = license_json['id'],
            name = license_json['name']
        )
        return [lic]

    def create_categories(self):
        """ Creates the "categories" portion of the COCO json
        Returns:
            categories: category objects that become part of the final json
            category_ids_by_name: a lookup dictionary for category ids based
                on the name of the category
        """
        cju = CategoryJsonUtils()
        categories = []
        category_ids_by_name = dict()
        category_id = 1 # 0 is reserved for the background

        super_categories = self.mask_definitions['super_categories']
        for super_category, _categories in super_categories.items():
            for category_name in _categories:
                categories.append(cju.create_coco_category(super_category, category_id, category_name))
                category_ids_by_name[category_name] = category_id
                category_id += 1

        return categories, category_ids_by_name

    def create_images_and_annotations(self, category_ids_by_name):
        """ Creates the list of images (in json) and the annotations for each
            image for the "image" and "annotations" portions of the COCO json
        """
        iju = ImageJsonUtils()
        aju = AnnotationJsonUtils()

        image_objs = []
        annotation_objs = []
        image_license = self.dataset_info['license']['id']
        image_id = 0

        mask_count = len(self.mask_definitions['masks'])
        print(f'Processing {mask_count} mask definitions...')

        # For each mask definition, create image and annotations
        for file_name, mask_def in tqdm(self.mask_definitions['masks'].items()):
            # Create a coco image json item
            image_path = Path(self.dataset_dir) / file_name
            image_obj = iju.create_coco_image(
                image_path,
                image_id,
                image_license)
            image_objs.append(image_obj)

            mask_path = Path(self.dataset_dir) / mask_def['mask']

            # Create a dict of category ids keyed by rgb_color
            category_ids_by_rgb = dict()
            for rgb_color, category in mask_def['color_categories'].items():
                category_ids_by_rgb[rgb_color] = category_ids_by_name[category['category']]
            annotation_obj = aju.create_coco_annotations(mask_path, image_id, category_ids_by_rgb)
            annotation_objs += annotation_obj # Add the new annotations to the existing list
            image_id += 1

        return image_objs, annotation_objs

    def main(self, args):
        self.validate_and_process_args(args)

        info = self.create_info()
        licenses = self.create_licenses()
        categories, category_ids_by_name = self.create_categories()
        images, annotations = self.create_images_and_annotations(category_ids_by_name)

        master_obj = {
            'info': info,
            'licenses': licenses,
            'images': images,
            'annotations': annotations,
            'categories': categories
        }

        # Write the json to a file
        output_path = Path(self.dataset_dir) / 'coco_instances.json'
        with open(output_path, 'w+') as output_file:
            json.dump(master_obj, output_file)

        print(f'Annotations successfully written to file:\n{output_path}')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate COCO JSON")

    parser.add_argument("-md", "--mask_definition", dest="mask_definition",
        help="path to a mask definition JSON file, generated by MaskJsonUtils module")
    parser.add_argument("-di", "--dataset_info", dest="dataset_info",
        help="path to a dataset info JSON file")

    args = parser.parse_args()

    cjc = CocoJsonCreator()
    cjc.main(args)
