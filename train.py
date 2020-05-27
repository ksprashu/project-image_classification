""" This program will train a Deep Learning model using Transfer Learning

    Sample Invocation:
    python train.py flowers --arch=alexnet --epochs=7 --gpu
"""

# organize imports
import argparse
import model_helper
import image_helper

# create argument parser and define inputs to the program
parser = argparse.ArgumentParser(description='Process inputs for training a Deep Learning model')
parser.add_argument('data_dir', action='store', help='Directory to data files')
parser.add_argument('--save_dir', action='store', default = '', help='Directory to save the trained model checkpoint to')
parser.add_argument('--arch', action='store', default='vgg13', help='Architecture of pre-trained Transfer Learning model to use')
parser.add_argument('--learning_rate', action='store', type=float, default=0.001, help='Learning rate to use for training the model')
parser.add_argument('--hidden_units', action='store', type=int, default=512, help='The size of the hidden layer to be used')
parser.add_argument('--epochs', action='store', type=int, default=3, help='The number of epochs or training iterations to be performed')
parser.add_argument('--gpu', action='store_true', default=False, help='Use the GPU for training and validation')

# parse input arguments
args = parser.parse_args()

# Now start the training process
# firstly get the image loaders
trainloader, validloader, testloader, class_to_idx = image_helper.get_image_loaders(args.data_dir)

# then get the custom model
model, optimizer = model_helper.build_custom_model(args.arch, args.hidden_units, args.learning_rate)

# get the device
device = model_helper.get_device(args.gpu)
print(f"using device '{device.type}'")

# perform the training and get the results
train_loss, valid_loss, valid_accuracy = model_helper.train_model(model, optimizer, args.epochs, trainloader, validloader, device)

# run the model against the test data 
test_loss, test_accuracy = model_helper.test_model(model, device, testloader)

# finally save the model checkpoint
model.class_to_idx = class_to_idx
model_helper.save_model_to_checkpoint(model, optimizer, args.save_dir)

print('Train program complete!')

      

