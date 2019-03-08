#!/usr/bin/env python3

import json
import warnings
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageEnhance

class MaskJsonUtils():
    """ Creates a JSON definition file for image masks.
    """

    def __init__(self, output_dir):
        """ Initializes the class.
        Args:
            output_dir: the directory where the definition file will be saved
        """
        self.output_dir = output_dir
        self.masks = dict()
        self.super_categories = dict()

    def add_category(self, category, super_category):
        """ Adds a new category to the set of the corresponding super_category
        Args:
            category: e.g. 'eagle'
            super_category: e.g. 'bird'
        Returns:
            True if successful, False if the category was already in the dictionary
        """
        if not self.super_categories.get(super_category):
            # Super category doesn't exist yet, create a new set
            self.super_categories[super_category] = {category}
        elif category in self.super_categories[super_category]:
            # Category is already accounted for
            return False
        else:
            # Add the category to the existing super category set
            self.super_categories[super_category].add(category)

        return True # Addition was successful

    def add_mask(self, image_path, mask_path, color_categories):
        """ Takes an image path, its corresponding mask path, and its color categories,
            and adds it to the appropriate dictionaries
        Args:
            image_path: the relative path to the image, e.g. './images/00000001.png'
            mask_path: the relative path to the mask image, e.g. './masks/00000001.png'
            color_categories: the legend of color categories, for this particular mask,
                represented as an rgb-color keyed dictionary of category names and their super categories.
                (the color category associations are not assumed to be consistent across images)
        Returns:
            True if successful, False if the image was already in the dictionary
        """
        if self.masks.get(image_path):
            return False # image/mask is already in the dictionary

        # Create the mask definition
        mask = {
            'mask': mask_path,
            'color_categories': color_categories
        }

        # Add the mask definition to the dictionary of masks
        self.masks[image_path] = mask

        # Regardless of color, we need to store each new category under its supercategory
        for _, item in color_categories.items():
            self.add_category(item['category'], item['super_category'])

        return True # Addition was successful

    def get_masks(self):
        """ Gets all masks that have been added
        """
        return self.masks

    def get_super_categories(self):
        """ Gets the dictionary of super categories for each category in a JSON
            serializable format
        Returns:
            A dictionary of lists of categories keyed on super_category
        """
        serializable_super_cats = dict()
        for super_cat, categories_set in self.super_categories.items():
            # Sets are not json serializable, so convert to list
            serializable_super_cats[super_cat] = list(categories_set)
        return serializable_super_cats

    def write_masks_to_json(self):
        """ Writes all masks and color categories to the output file path as JSON
        """
        # Serialize the masks and super categories dictionaries
        serializable_masks = self.get_masks()
        serializable_super_cats = self.get_super_categories()
        masks_obj = {
            'masks': serializable_masks,
            'super_categories': serializable_super_cats
        }

        # Write the JSON output file
        output_file_path = Path(self.output_dir) / 'mask_definitions.json'
        with open(output_file_path), 'w+') as json_file:
            json_file.write(json.dumps(masks_obj))

class ImageComposition():
    """ Composes images together in random ways, applying transformations to the foreground to create a synthetic
        combined image.
    """

    def __init__(self):
        self.allowed_output_types = ['.png', '.jpg', '.jpeg']
        self.allowed_background_types = ['.png', '.jpg', '.jpeg']
        self.zero_padding = 8 # 00000027.png, supports up to 100 million images

    def _validate_and_process_args(self, args):
        # Validates input arguments and sets up class variables
        # Args:
        #     args: the ArgumentParser command line arguments

        # Validate the count
        assert args.count > 0, 'count must be greater than 0'
        self.count = args.count

        # Validate the width and height
        assert args.width >= 64, 'width must be greater than 64'
        self.width = args.width
        assert args.height >= 64, 'height must be greater than 64'
        self.height = args.height

        # Validate and process the output type
        if args.output_type is None:
            self.output_type = '.jpg' # default
        else:
            if args.output_type[0] != '.':
                self.output_type = f'.{args.output_type'
            assert self.output_type in self.allowed_output_types, f'output_type is not supported: {self.output_type}'

        # Validate and process output and input directories
        self._validate_and_process_output_directory()
        self._validate_and_process_input_directory()

    def _validate_and_process_output_directory(self):
        self.output_dir = Path(args.output_dir)
        self.images_output_dir = self.output_dir / 'images'
        self.masks_output_dir = self.output_dir / 'masks'

        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.images_output_dir.mkdir(exist_ok=True)
        self.masks_output_dir.mkdir(exist_ok=True)

        # Check for existing contents in the images directory
        for _ in self.images_output_dir.iterdir():
            # We found something, check if the user wants to overwrite files or quit
            should_continue = input('output_dir is not empty, files may be overwritten.\nContinue (y/n)? ').lower()
            if should_continue != 'y' and should_continue != 'yes':
                quit()
            break

    def _validate_and_process_input_directory(self):
        self.input_dir = Path(args.input_dir)
        assert self.input_dir.exists(), f'input_dir does not exist: {args.input_dir}'

        for x in self.input_dir.iterdir():
            if x.name == 'foregrounds':
                self.foregrounds_dir = x
            elif x.name == 'backgrounds':
                self.backgrounds_dir = x

        assert self.foregrounds_dir is not None, 'foregrounds sub-directory was not found in the input_dir'
        assert self.backgrounds_dir is not None, 'backgrounds sub-directory was not found in the input_dir'

        self._validate_and_process_foregrounds()
        self._validate_and_process_backgrounds()

    def _validate_and_process_foregrounds(self):
        # Validates input foregrounds and processes them into a foregrounds dictionary.
        # Expected directory structure:
        # + foregrounds_dir
        #     + super_category_dir
        #         + category_dir
        #             + foreground_image.png

        self.foregrounds_dict = dict()

        for super_category_dir in self.foregrounds_dir.iterdir():
            if not super_category_dir.is_dir():
                warnings.warn(f'file found in foregrounds directory (expected super-category directories), ignoring: {super_category_dir}')
                continue

            # This is a super category directory
            for category_dir in super_category_dir.iterdir():
                if not category_dir.is_dir():
                    warnings.warn(f'file found in super category directory (expected category directories), ignoring: {category_dir}')
                    continue

                # This is a category directory
                for image_file in category_dir.iterdir():
                    if not image_file.is_file():
                        warnings.warn(f'a directory was found inside a category directory, ignoring: {str(image_file)}')
                        continue
                    if image_file.suffix != '.png':
                        warnings.warn(f'foreground must be a .png file, skipping: {str(image_file)}')
                        continue

                    # Valid foreground image, add to foregrounds_dict
                    super_category = super_category_dir.name
                    category = category_dir.name

                    if super_category not in self.foregrounds_dict:
                        self.foregrounds_dict[super_category] = dict()

                    if category not in self.foregrounds_dict[super_category]:
                        self.foregrounds_dict[super_category][category] = []

                    self.foregrounds_dict[super_category][category].append(image_file)

        assert len(self.foregrounds_dict) > 0, 'no valid foregrounds were found'

    def validate_and_process_backgrounds(self):
        self.backgrounds = []
        for image_file in self.backgrounds_dir.iterdir():
            if not image_file.is_file():
                warnings.warn(f'a directory was found inside the backgrounds directory, ignoring: {image_file}')
                continue

            if image_file.suffix not in self.allowed_background_types:
                warnings.warn(f'background must match an accepted type {str(self.allowed_background_types)}, ignoring: {image_file}')
                continue

            # Valid file, add to backgrounds list
            self.backgrounds.append(image_file)

        assert len(self.backgrounds) > 0, 'no valid backgrounds were found'

    def _generate_images(self):
        # Generates a number of images and creates segmentation masks, then
        # saves a mask_definitions.json file that describes the dataset.

        print(f'Generating {self.count} images with masks...')

        mju = MaskJsonUtils(self.output_dir)

        # Create all images/masks (with tqdm to have a progress bar)
        for i in tqdm(range(self.count)):
            # Randomly choose a foreground and background
            background_path = random.choice(self.backgrounds)
            super_category = random.choice(list(self.foregrounds_dict.keys()))
            category = random.choice(list(self.foregrounds_dict[super_category].keys()))
            foreground_path = random.choice(self.foregrounds_dict[super_category][category])

            # Compose foreground and background
            mask_rgb_color = (255, 0, 0)
            composite, mask = self._compose_images(foreground_path, background_path, mask_rgb_color)

            # Create the file name (used for both composite and mask)
            save_filename = f'{i:0{self.zero_padding}}' # e.g. 00000023.jpg

            # Save composite image to the images sub-directory
            composite_filename = f'{save_filename}{self.output_type}' # e.g. 00000023.jpg
            composite_path = self.output_dir / 'images' / composite_filename # e.g. my_output_dir/images/00000023.jpg
            composite = composite.convert('RGB') # remove alpha
            composite.save(composite_path)

            # Save the mask image to the masks sub-directory
            mask_filename = f'{save_filename}.png' # masks are always png to avoid lossy compression
            mask_path = self.output_dir / 'masks' / mask_filename # e.g. my_output_dir/masks/00000023.png
            mask.save(mask_path)

            # Add category and mask to MaskJsonUtils
            mju.add_category(category, super_category)
            color_categories = {str(mask_rgb_color):{'category':category, 'super_category':super_category}}
            mju.add_masks(
                composite_path.relative_to(self.output_dir).as_posix(),
                mask_path.relative_to(self.output_dir).as_posix(),
                color_categories
            )

        #Write masks to json
        mju.write_masks_to_json()

    def _compose_images(self, foreground_path, background_path, mask_rgb_color):
        # Composes a foreground image and a background image and creates a segmentation mask
        # using the specified color. Validation should already be done by now.
        # Args:
        #     foreground_path: the path to a valid foreground image
        #     background_path: the path to a valid background image
        #     mask_rgb_color: the desired foreground color for the mask
        # Returns:
        #     composite: the composed image
        #     mask: the mask image

        # Open foreground and get the alpha channel
        foreground = Image.open(foreground_path)
        foreground_alpha = np.array(foreground.getchannel(3))
        assert np.any(foreground_alpha == 0), f'foreground needs to have some transparency: {str(foreground_path)}'

        # Open background and convert to RGBA
        background = Image.open(background_path)
        background = background.convert('RGBA')

        # ** Apply Transformations **
        # Rotate the foreground
        angle_degrees = random.randint(0, 359)
        foreground = foreground.rotate(angle_degrees, resample=Image.BICUBIC, expand=True)

        # Scale the foreground
        scale = random.random() * .5 + .5 # Pick something between .5 and 1
        new_size = (int(foreground.size[0] * scale), int(foreground.size[1] * scale))
        foreground = foreground.resize(new_size, resample=Image.BICUBIC)

        # Adjust foreground brightness
        brightness_factor = random.random() * .4 + .7 # Pick something between .7 and 1.1
        enhancer = ImageEnhance.Brightness(foreground)
        foreground = enhancer.enhance(brightness_factor)

        # Add any other transformations here...

        # Crop background to desired size (self.width x self.height), randomly positioned
        bg_width, bg_height = background.size
        max_crop_x_pos = bg_width - self.width
        max_crop_y_pos = bg_height - self.height
        assert max_crop_x_pos >= 0, f'desired width, {self.width}, is greater than background width, {bg_width}, for {str(background_path)}'
        assert max_crop_y_pos >= 0, f'desired height, {self.height}, is greater than background height, {bg_height}, for {str(background_path)}'
        crop_x_pos = random.randint(0, max_crop_x_pos)
        crop_y_pos = random.randint(0, max_crop_y_pos)
        background = background.crop((crop_x_pos, crop_y_pos, crop_x_pos + self.width, crop_y_pos + self.height))

        # Create a new foreground image as large as the background and paste it on top
        new_foreground = Image.new('RGBA', background.size, color = (0, 0, 0, 0))
        new_foreground.paste(foreground, paste_position)

        # Extract the alpha channel from the foreground and paste it into a new image the size of the background
        alpha_mask = foreground.getchannel(3)
        new_alpha_mask = Image.new('L', background.size, color = 0)
        new_alpha_mask.paste(alpha_mask, paste_position)
        composite = Image.composite(new_foreground, background, new_alpha_mask)

        # Grab the alpha pixels above a specified threshold
        alpha_threshold = 200
        mask_arr = np.array(np.greater(np.array(new_alpha_mask), alpha_threshold), dtype=np.uint8)
        uint8_mask = np.uint8(mask_arr) # This is composed of 1s and 0s

        # Multiply the mask value (1 or 0) by the color in each RGB channel and ocmbine to get the mask
        red_channel = uint8_mask * mask_rgb_color[0]
        green_channel = uint8_mask * mask_rgb_color[1]
        blue_channel = uint8_mask * mask_rgb_color[2]
        rgb_mask_arr = np.dstack((red_channel, green_channel, blue_channel))
        mask = Image.fromarray(rgb_mask_arr, 'RGB')

        return composite, mask

    # Start here
    def main(self, args):
        self._validate_and_process_args(args)
        self._generate_images()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image Composition")
    parser.add_argument("--input_dir", type=str, dest="input_dir", required=True, help="The input directory. \
                        This contains a 'backgrounds' directory of pngs or jpgs, and a 'foregrounds' directory which \
                        contains supercategory directories (e.g. 'animal', 'vehicle'), each of which contain category \
                        directories (e.g. 'horse', 'bear'). Each category directory contains png images of that item on a \
                        transparent background (e.g. a grizzly bear on a transparent background).")
    parser.add_argument("--output_dir", type=str, dest="output_dir", required=True, help="The directory where images, masks, \
                        and json files will be placed")
    parser.add_argument("--count", type=int, dest="count", required=True, help="number of composed images to create")
    parser.add_argument("--width", type=int, dest="width", required=True, help="output image pixel width")
    parser.add_argument("--height", type=int, dest="height", required=True, help="output image pixel height")
    parser.add_argument("--output_type", type=str, dest="output_type", help="png or jpg (default)")

    args = parser.parse_args()

    image_comp = ImageComposition()
    image_comp.main(args)