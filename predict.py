""" This program will take an input of a flower image and predict its name

    Sample invocation:
    python predict.py flowers/test/37/image_03811.jpg checkpoint.pth --category_name cat_to_name.json --gpu
"""

# organize imports
import matplotlib.pyplot as plt
import image_helper
import model_helper
import argparse

# create argument parser and define inputs to the program
parser = argparse.ArgumentParser(description='Process inputs for predicting on an image file')
parser.add_argument('image_dir', action='store', help='Full path to the image for prediction')
parser.add_argument('checkpoint', action='store', default='', help='Path to and name of the checkpoint to be used to load the model')
parser.add_argument('--top_k', action='store', default=3, type=int, help='Number of top results from the prediction to be returned')
parser.add_argument('--category_names', action='store', default='', help='Name of the json file from which to retrieve categories')
parser.add_argument('--gpu', action='store_true', default=False, help='Use the GPU for the prediction')

# parse input arguments
args = parser.parse_args()

# Now start the prediction process
# get the device where to run on
device = model_helper.get_device(args.gpu)

# Load model from checkpoint
model, optimizer = model_helper.load_model_from_checkpoint(args.checkpoint, device)

# Load image for prediction and do prediction
image = image_helper.process_image(args.image_dir)
topk_ps, topk_classes = model_helper.predict(image, model, device, args.top_k)
    
# list out the topk classes that were predicted and their class names if available
print(f"the top {args.top_k} likely classes for this image along with probabilities are:")
if not args.category_names == '':
    topk_classnames = image_helper.get_cat_names(topk_classes, args.category_names)
    for name, cat, ps in zip(topk_classnames, topk_classes, topk_ps):
        print(f"\t({cat}) {name:.15s}\t: {ps:.3f}")
else:
    for cat, ps in zip(topk_classes, topk_ps):
        print(f"\t{cat}\t: {ps:.3f}")



