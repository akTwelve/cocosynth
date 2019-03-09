# Getting Started with COCO Synth

## Initial Setup
It is recommended that you use Anaconda for Python environment management. It will help you install Shapely, which doesn't work when you just try to install it with pip.

```
conda create -n cocosynth python=3.7
activate cocosynth
conda install -c conda-forge shapely
pip install -r requirements.txt
```

# Create Synthetic Images and Masks
In this section, we will use "image_composition.py" to take randomly pick foregrounds and automatically super-impose them on backgrounds. You will need a number of foreground cutouts with transparent backgrounds. For example, you might have a picture of an eagle with a completely transparent background. Due to the need for transparency, these images should be .png format (.jpg doesn't have transparency).

Directory setup:
- Inside the "datasets" directory, create a new folder for your dataset (e.g. "wild_animal_dataset")
- Inside that dataset directory, create a folder called "input"
- Inside "input", create two folders called "foregrounds" and "backgrounds"
- Inside "foregrounds", create a folder for each super category (e.g. "bird", "lizard")
- Inside each foreground super category folder, create a folder for each category (e.g. "eagle", "owl")
- Inside each category folder, add all foreground photos you intend to use for the respective category (e.g. all of you eagle foreground cutouts)
- Inside "backgrounds", add all background photos you intend to use

Run "image_composition.py" to create your images and masks
```
python ./python/image_composition.py --input_dir ./datasets/box_dataset_synthetic/input --output_dir ./datasets/box_dataset_synthetic/output --count 10 --width 512 --height 512
```

# Create COCO Instances JSON
Now we're going to use the images, masks, and json to create COCO instances.

Optional: Run "coco_json_utils.py" with --help to see the documentation. This will explain the next command.
```
python ./python/coco_json_utils.py --help
```
Run the command with the correct parameters
```
python ./python/coco_json_utils.py -md ./datasets/box_dataset_synthetic/output/mask_definitions.json -di ./datasets/box_dataset_synthetic/output/dataset_info.json
```

You will now have a new json file called "coco_instances.json". This is contains all of your COCO json!

## View Segmentations with the COCO Image Viewer
Fire up Jupyter Notebook and open up "coco_image_viewer.ipynb" in your browser.
```
jupyter notebook
```

Run through the cells in the notebook. As long as you have the files in the correct places, you should see annotations at the end.

