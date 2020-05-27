""" This is a helper module to support with loading and pre-processing of images
"""

# organize imports
import torch
from torchvision import datasets, transforms
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_image_loaders(data_dir):
    """This method returns the training, validation, and testing loaders
        
    The method expects to find 3 sub directories under the data directory
    called 'train', 'valid', 'test' where it will find the images samples
    for training, validation, and testing respectively
        
    It will then read and apply transformations to the datasets and finally
    create 3 loaders for each of the datasets and return it back.
        
    Args:
    data_dir: string. The directory path to where the images can be found
    
    Returns:
    trainloader, validloader, testloader. A tuple of the 3 loaders based on
    the 3 datasets
    """
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define the transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomGrayscale(p=0.1),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the transforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    return trainloader, validloader, testloader, train_dataset.class_to_idx
  
        
def get_cat_names(cat_classes, cat_file):
    """Return the cateory names for the list of provided classes
    
    Args:
    cat_classes: List<string>. Array of classes for which to retrieve the category names
    cat_file: string. The path of the file that has the category to name mapping
    """

    # open the file
    with open(cat_file, 'r') as f:
        cat_to_name = json.load(f)
        
    # get cateogry names in the same sequence as provided classes
    cat_names = [cat_to_name[c] for c in cat_classes] 
    
    return cat_names
    
    
def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model, returns a Numpy array
    
    Args:
    image: PIL image to be processed
    
    Returns:
    Tensor of the processed image
    """
    
    # Load the image
    image = Image.open(image)

    # Process a PIL image for use in a PyTorch model
    image = image.resize((255,255))
    image = image.crop((16,16,240,240))
    
    # get the image as numpy array and put the color channel in front
    np_image = np.array(image).transpose((2,0,1))
    
    # minimum of each color 
    mins = np_image.min(axis=(1,2)).reshape(3,1,1)
    # maximum of each color 
    maxes = np_image.max(axis=(1,2)).reshape(3,1,1)
    
    # normalize the images
    np_image = (np_image - mins) / (maxes - mins)

    # expected means and standard deviation
    mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
    
    # standardize the color channels
    np_image = (np_image - mean) / std
    image_tensor = torch.from_numpy(np_image)

    return image_tensor.float()
    
        
def imshow(image, ax=None, title=None):
    """Displays image based on given tensor
    
    Args: 
    image: the image represented by a tensor
    ax: the axes on which to draw the image
    title: the title for the image
    
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    if title:
        ax.set_title(title)
    
    return ax    
    
    
    