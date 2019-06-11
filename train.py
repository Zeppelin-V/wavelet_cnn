import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
import numpy as np
import cnn
import img_dataloader


def get_model(model_name, computing_device, wvlt_transform):

    if (model_name == "BaselineCNN"):
        x = 0
    elif (model_name == "WaveletCNN"):
        model= cnn.waveletCNN(wvlt_transform)
        model = model.to(computing_device)
        print("Model on CUDA?", next(model.paramaters()).is_cuda)
        return model

    return 0


def train_model(model_name, computing_device, val_indices, epochs, k, learning_rate,
                batch_size, num_minibatches, wvlt_transform, transform, extras):


    #List of validation/train losses per minibatch for each validation
    all_val_total_loss = []
    all_train_total_loss = []
    all_val_avg_minibatch_loss = []
    all_train_avg_minibatch_loss = []
    all_train_acc = []
    all_val_acc = []


    #Define the loss criterion and instantiate the gradient descent optimizer
    criterion = nn.CrossEntropyLoss()

    #Iterate through every possible validation/training split
    for i in range(0, k):

        #Get a new model for this validation/training split
        model = get_model(model_name, computing_device, wvlt_transform)

        #Instantiate the gradient descent optimizer - use Adam optimizaer with default parameters
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        #Load validation, training data
        val_loader, train_loader = img_dataloader.create_k_split_dataloaders(val_indices,
                                                                             batch_size, i, transform,
                                                                             show_sample=False, extras=extras)

        #Make lists for total and avg_minibatch loss for validation and training
        val_epoch_loss = []
        train_epoch_loss = []
        avg_train_mb_epoch_loss = []
        avg_val_mb_epoch_loss = []
        val_epoch_acc = []
        train_epoch_acc = []


        #Train for x epochs
        for epoch in range(0, epochs):

            val_total_loss = []
            train_total_loss = []
            avg_train_minibatch_loss = []
            avg_val_minibatch_loss = []
            val_accuracy = []
            train_accuracy = []


            N = num_minibatches
            N_minibatch_loss = 0.0

            print("Training model on training set. i = ", i, " epoch num = ", epoch)

            #Run train data through model
            for minibatch_count, (images, labels) in enumerate(train_loader):


                #Put the minibatch data in CUDA Tensors and run on the GPU if supported
                images, labels = images.to(computing_device), labels.to(computing_device)

                #Zero out the stored gradient (buffer) from the previous iteration
                optimizer.zero_grad()

                #Perform the forward pass through the network and compute the loss
                output = model(images)
                loss_tensor = criterion(output, labels)

                #Backpropogate XD
                loss_tensor.backward()

                #Update the weights
                optimizer.step()

                # Add this iteration's loss to the total loss
                train_total_loss.append(loss_tensor.item())
                N_minibatch_loss += loss_tensor

                #Calculate avg_minibatch loss
                if minibatch_count % N == 0:

                    #Print the loss averaged over the last N mini-batches
                    N_minibatch_loss /= N
                    print("Epoch ", str(epoch + 1), " average minibatch # ", str(minibatch_count),
                          "loss: ", str(N_minibatch_loss))

                    #Add the averaged loss over N minibatches and reset the counter
                    avg_train_minibatch_loss.append(N_minibatch_loss)

                    N_minibatch_loss = 0.0

                    train_acc = (torch.sum(labels.eq(output), dim=0).cpu().long())
                    avg_acc = torch.mea((train_acc.to(dtype=torch.float) / (len(output))).float())
                    train_accuracy.append(avg_acc)


            #Add this to the list of average training minibatch loss to list for all val/training splits
            train_epoch_loss.append(train_total_loss)
            avg_train_mb_epoch_loss.append(avg_train_minibatch_loss)
            train_epoch_acc.append(train_accuracy)

            #Test model on validation set
            N_minibatch_loss = 0.0

            print("Testing model on ", i, "the validation set")

            #Run through minibatches
            for minibatch_count, (images, labels) in enumerate(val_loader, 0):

                #Put the minibatch data in CUDA Tensors and run on the GPU if supported
                images, labels = images.to(computing_device), labels.to(computing_device)

                #Compute output with no gradients to reduce memory consumption
                with torch.no_grad():
                    output = model(images)

                loss_tensor = criterion(output, labels)

                #Add loss to our loss list
                val_total_loss.append(loss_tensor.item())
                N_minibatch_loss += loss_tensor


                #Calculate avg minibatch loss
                if minibatch_count % N == 0:

                    #Print the loss averaged over the last N mini-batches
                    N_minibatch_loss /= N
                    print("Epoch ", str(epoch+1), " average minibatch #", minibatch_count, " loss: ", N_minibatch_loss)

                    #Add the averaged loss over N minibatches and reset the counter
                    avg_val_minibatch_loss.append(N_minibatch_loss)

                    N_minibatch_loss = 0.0

                    val_acc = (torch.sum(labels.eq(output), dim=0).cpu().long())
                    avg_acc = torch.mean((val_acc.to(dtype=torch.float) / (len(output))).float())
                    val_accuracy.append(avg_acc)


            #Add loss lists for validation data to list of all possible losses
            val_epoch_loss.append(val_total_loss)
            avg_val_mb_epoch_loss.append(avg_val_minibatch_loss)
            val_epoch_acc.append(val_accuracy)


        #Record this train/validation splits loss/accuracy metrics
        all_val_total_loss.append(val_epoch_loss)
        all_train_total_loss.append(train_epoch_loss)
        all_val_avg_minibatch_loss.append(avg_val_mb_epoch_loss)
        all_train_avg_minibatch_loss.append(avg_train_mb_epoch_loss)
        all_train_acc.append(train_epoch_acc)
        all_val_acc.append(val_epoch_acc)


    return (all_val_total_loss, all_train_total_loss, all_val_avg_minibatch_loss, all_train_avg_minibatch_loss,
            all_val_acc, all_train_total_loss, all_train_avg_minibatch_loss, all_train_acc)




def test_model(model_name, computing_device, test_loader, train_inidices, epochs, learning_rate, batch_size,
               num_minibatches, wvlt_transform, transform, extras):

    #Define the loss criterion and instantiate the gradient descent optimizer
    criterion = nn.CrossEntropyLoss()

    #Get a new model for this training split
    model = get_model(model_name, computing_device, wvlt_transform)

    #Instantiate the gradient descent optimizer - use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #Get training dataloader
    train_loader = img_dataloader.create_train_dataloader(train_inidices, batch_size, transform,
                                                          show_sample=False, extras=extras)

    #Record train_loss and avg_train_mb loss
    train_total_loss = []
    avg_train_minibatch_loss = []

    #Train for x epochs
    for epoch in range(0, epochs):

        N = num_minibatches
        N_minibatch_loss = 0.0

        print("Training model on training set for epoch num: ", epoch)

        #Run train data through model
        for minibatch_count, (images, labels) in enumerate(train_loader):

            #Put the minibatch data in CUDA Tensors and run on the GPU if possible
            images, labels = images.to(computing_device), labels.to(computing_device)

            #Zero out the stored gradient (buffer) from the previous iteration
            optimizer.zero_grad()

            #Perform the forward pass through the network and compute the loss
            outputs = model(images)
            loss = criterion(outputs, labels)

            #Backprop XD
            loss.backward()

            #Update the weights
            optimizer.step()

            #Add this iteration's loss to the total_loss
            train_total_loss.append(loss.item())
            N_minibatch_loss += loss

            if minibatch_count % N == 0:

                #Print the loss averaged over the last N mini-batches
                N_minibatch_loss /= N
                print("Epoch ", str(epoch+1), " avg minibatch # ", minibatch_count, " loss: ", N_minibatch_loss)


                #Add the averaged loss over N_minibatches and reset the counter
                avg_train_minibatch_loss.append(N_minibatch_loss)

                N_minibatch_loss = 0.0


        N_minibatch_loss = 0.0

        test_total_loss = []
        test_avg_minibatch_loss = []

        #Iterate through testing minibatches
        for minibatch_count, (images,labels) in enumerate(test_loader, 0):

            #Get labels and images on CUDA
            images, labels = images.to(computing_device), labels.to(computing_device)

            #Forward pass with no gradients to save memory
            with torch.no_grad():
                output = model(images)
            loss = criterion(output, labels)

            #Add this iteration's loss to the total_loss
            test_total_loss.append(loss.item())
            N_minibatch_loss += loss

            if minibatch_count % N == 0:
                #Print the loss averaged over the last N minibatches
                print("Epoch ", str(epoch + 1), " average minibatch # ", minibatch_count, " loss: ", N_minibatch_loss)

                #Add the averaged loss over N minibatches and reset the counter
                test_avg_minibatch_loss.append(N_minibatch_loss)

                N_minibatch_loss = 0.0


            #TODO: Implement the accuracy/loss metrics for testing the model

            return 0




