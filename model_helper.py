""" This module defines the custom neural network based on a pre-trained model
"""

# organize imports
from torchvision import models
import sys
import torch
from torch import nn
from torch import optim
from workspace_utils import active_session
import os


def build_custom_model(arch, hidden_units, learning_rate):
    """Builds and returns a custom model based on an existing pre-trained model
    
    The custom model will be used for transfer learning, and will be based
    on the model architecture that is passed in.
    
    Args:
    arch: string. The architecture / name of the pre-trained model
    hidden_units: integer. The number of untils in the hidden layers
    learning_rate: float. The rate of learning for updating the weights
    
    Returns:
    model. The fully connected model to be used
    """
    
    OUTPUT_UNITS = 102
    
    try:
        # dynamically get the model based on the passed architecture
        trained_model = getattr(models, arch)
    except:
        error_str = f"Model {arch} does not exist. Aborting!"
        sys.exit(error_str)
        
    # get the pretrained model    
    model = trained_model(pretrained=True)
    
    # freeze the weights of the convolutional layers
    for param in model.parameters():
        param.requires_grad = False
    
    # get the name of the last child which is the classifier
    for child_name, child_module in model.named_children():
        pass
    classifier = getattr(model, child_name)

    # get the first layer of the classifier which has the in_features attribute
    for name, layer in classifier.named_children():
        if hasattr(layer, 'in_features'):
            break

    # or the classifier itself in case there are no children
    else:
        layer = classifier 
        
    # check that the hidden_inputs is in a valid range
    INPUT_UNITS = layer.in_features  
    if hidden_units > INPUT_UNITS or hidden_units < OUTPUT_UNITS:
        error_str = f"hidden units should be between {INPUT_UNITS} and {OUTPUT_UNITS}"
        sys.exit(error_str)

    # build the final classifier
    new_fc = nn.Sequential(nn.Linear(INPUT_UNITS, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(hidden_units, OUTPUT_UNITS),
                               nn.LogSoftmax(dim=1))

    # assign new classifier back to model
    setattr(model, child_name, new_fc)
    
    # store data back into model
    model.arch = arch
    model.hidden_units = hidden_units
    model.learning_rate = learning_rate
    model.classifier_name = child_name    
    
    # build an Adam optimizer
    optimizer = optim.Adam(new_fc.parameters(), lr=learning_rate)
    
    return model, optimizer


def get_device(gpu):
    """ Return the device available for compute
    """
    
    # set to cpu by default
    device = torch.device('cpu')
    
    # if gpu was requested, check if available and then pass back
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device


def __validate_model(model, device, validloader, batch_size=999):
    """ returns the loss and accuracy after running validation iterations on a model and dataset
    
    Args:
    model: the model that should be used to feed forward
    validloader: the dataset load that should be used to get images and labels
    batch_size: integer - the # of batches from loader to use 
    device: the torch device on which to run the model on
    
    
    Returns:
    loss: validation loss after validating the specified batches
    accuracy: the accuracy of prediction on the validation set
    """
    
    with torch.no_grad(): # turn off gradient calculations
        model.eval() # turn off dropout
        val_step = 0
        val_loss = 0
        accuracy = 0
        criterion = nn.NLLLoss()

        for images, labels in validloader: # images in validation set
            val_step += 1

            images, labels = images.to(device), labels.to(device) # pushing to available device

            log_ps = model.forward(images) # feed forward
            ps = torch.exp(log_ps) # convert log to actual probabilities
            top_ps, top_pred = ps.topk(1, dim=1) # get the top probability and its index

            matches = top_pred == labels.view(*top_pred.shape) # compare prediction with actual labels
            accuracy += torch.mean(matches.type(torch.float)) # compute and accumulate accuracy for this batch

            batch_loss = criterion(log_ps, labels) # calculte loss in this run
            val_loss += batch_loss.item() # accumulate validation losses

            if val_step >= batch_size: # don't want to go through all the validations cases each time
                break

    model.train() # turn dropout back on
    
    return val_loss, accuracy

    
def train_model(model, optimizer, epochs, trainloader, validloader, device):
    """ Train the passed model and return associated losses and accuracy
    
    Args:
    model: The model which has to be trained
    optimzer: The optimizer to be used for updating the weights
    epochs: integer. The number of training iterations to be run
    trainloader: The loader for the training images
    validloader: The loader for the validation images
    device: The pytorch device on which to run the model
    
    Returns:
    training_loss: float. The total loss while predicting on the training set
    validation_loss: float. The total loss while predicting on the valiation set
    validation_accuracy: float. How accurate was the model in predictions on the complete validation set
    """
    
    step_check = 5 # check every x steps
    val_batches = 5 # use these many batches in validation runs
    running_loss = 0 # keeping track of losses

    # keep alive till all iterations are done
    with active_session():
        
        # define the criterion
        criterion = nn.NLLLoss()
        
        # push model to device
        model.to(device)

        # run training iterations
        print(f"training with {epochs} iterations")
        for epoch in range(epochs):
            step = 0 # track training steps
            train_loss = 0 # track total loss

            for images, labels in trainloader: # images in training set
                step += 1 

                optimizer.zero_grad() # reset the gradient on each iteration
                images, labels = images.to(device), labels.to(device) # push images and labels to device

                log_ps = model.forward(images) # feed the input forward
                loss = criterion(log_ps, labels) # calcuate loss

                loss.backward() # backprop the weights
                optimizer.step() # adjust weights by a step

                running_loss += loss.item() # accumulate the loss
                train_loss += running_loss # accumulate training losses

                # every x steps, compare the training progress against the validation step
                if step % step_check == 0:
                    
                    # run the validation pipeline
                    val_loss, accuracy = __validate_model(model, device, validloader, val_batches) 
                    
                    # print the current status of training
                    print(f"\tEpoch = {epoch+1}/{epochs}, "
                          f"Step = {step}/{len(trainloader)}, "
                          f"Train loss = {running_loss / step_check:.3f}, "
                          f"Val loss = {val_loss / val_batches:.3f}, "
                          f"Val accuracy = {accuracy / val_batches:.3f}")

                    running_loss = 0 # reset running loss for next batch
                    
            else: # once the trainloader loop ends, then do a full validation run

                # run the validation pipeline with the entire batch size
                val_loss, accuracy = __validate_model(model, device, validloader) 

                # print the current status of training
                training_loss = train_loss / len(trainloader)
                validation_loss = val_loss / len(validloader)
                validation_accuracy = accuracy / len(validloader)

                print(f"Completed Epoch = {epoch+1}/{epochs}, "
                      f"Train loss = {training_loss:.3f}, "
                      f"Val loss = {validation_loss:.3f}, "
                      f"Val accuracy = {validation_accuracy:.3f}\n")
    
    # save essential data into model
    model.train_loss = training_loss
    model.device = device.type
    model.epochs = epochs      
    
    return training_loss, validation_loss, validation_accuracy
        
        
def test_model(model, device, testloader):
    """Run the test data on the model and return results
    
    Args:
    model: The model which has to be used for testing
    device: The torch device on which to run the model on
    testloader: The image loader for testing data    
    """

    test_loss, test_accuracy = __validate_model(model, device, testloader)
    print(f"Test Accuracy = {test_accuracy / len(testloader):.3f}")    

    return test_loss, test_accuracy


def save_model_to_checkpoint(model, optimizer, save_dir):
    """Save the model checkpoint
    
    Args:
    model: the model to be saved
    optimizer: the optimizer to be saved
    save_dir: string. the path into which to save the model
    """
    
    # do all save on cpu for compatibility
    device = model.device
    if not device == 'cpu':
        model = model.to(torch.device('cpu'))
    
    # Save the checkpoint 
    checkpoint = {'arch': model.arch,
                  'device': model.device,
                  'learning_rate': model.learning_rate,
                  'hidden_units': model.hidden_units,
                  'epochs': model.epochs,                  
                  'train_loss': model.train_loss,
                  'class_to_idx': model.class_to_idx,
                  'classifier_name': model.classifier_name,
                  'classifier': getattr(model, model.classifier_name),
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}
    
    # use the provided directory if available
    filepath = 'checkpoint.pth' if save_dir == '' else save_dir
    torch.save(checkpoint, filepath)
    print(f"checkpoint saved at {filepath}")
    
    # restore the model back to original device
    if not device == 'cpu':
        model = model.to(torch.device(device))
        
    return

    
def load_model_from_checkpoint(checkpoint, device):
    """Takes the name of a checkpoint file and returns the re-created model and optimizer
        
    Args:
    save_dir: string - directory for checkpoint file
    device: torch device for loading the model
    """
    
    # load the checkpoint 
    checkpoint = torch.load(checkpoint)
    
    # rebuild the custom model
    model, optimizer = build_custom_model(checkpoint['arch'], checkpoint['hidden_units'], checkpoint['learning_rate'])
        
    # restore essential data back into the model
    model.class_to_idx = checkpoint['class_to_idx']
    model.epochs = checkpoint['epochs']
    model.train_loss = checkpoint['train_loss']
    model.device = checkpoint['device']

    # replace classifier and classifier state_dict back into the model
    model.classifier_name = checkpoint['classifier_name']
    setattr(model, checkpoint['classifier_name'], checkpoint['classifier'])
    model.load_state_dict(checkpoint['model_state_dict'])

    # restore optimizer state_dict 
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # ready the model by loading to device and turning on evaluation mode
    if not device == 'cpu':
        model.to(torch.device(device))
        
    model.eval()

    return model, optimizer


def predict(image, model, device, topk=5, display=False):
    """ Predict the class (or classes) of an image using a trained deep learning model.
    
    Args:
    image: the image to be used for prediction
    model: the trained model to use
    topk: the number of top results to be returned
    """
    
    # Implement the code to predict the class from an image file
    image_tensor = image.view(1, *image.shape) # convert to list
    
    # push model to device
    model.to(device)

    # firstly turn model in evaluation mode without dropout
    model.eval()
    with torch.no_grad():
        # now feed image forward and get the log outputs
        image_tensor = image_tensor.to(device)
        log_ps = model.forward(image_tensor)
        
        # convert log output to actual probilities and get topk probabilities
        ps = torch.exp(log_ps)
        topk_ps, topk_indicies = ps.topk(topk, dim=1)
    
    # convert output tensors into 1D numpy arrays
    topk_ps = topk_ps.cpu()
    topk_ps = topk_ps.view(topk).numpy()
    topk_indicies = topk_indicies.cpu()
    topk_indicies = topk_indicies.view(topk).numpy()

    # get category classes from index
    topk_classes = []
    dict_keys = list(model.class_to_idx.keys())
    dict_vals = list(model.class_to_idx.values())
    for idx in topk_indicies:
        topk_classes.append(dict_keys[dict_vals.index(idx)])

    # display the image
    if display:
        fig, (ax1,ax2) = plt.subplots(2,1, figsize=(5,10))
        imshow(image_tensor[0], ax=ax1, title=cat_names[0])

        ax1.axis('off') # don't show ticks for image

        # display the probability classes
        ax2.barh(np.arange(topk_ps.shape[0]), list(topk_ps), align="center")

        # set labels / ticks / titles
        ax2.set_yticks(np.arange(topk_ps.shape[0]))
        ax2.set_yticklabels(cat_names)
        ax2.set_xlabel('Probabilities')
        ax2.set_ylabel('Labels')
        ax2.set_title('Predictions')
        ax2.invert_yaxis() # biggest class shows on top

        # set layout and display
        plt.show()
    
    return topk_ps, topk_classes

   