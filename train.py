import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn

from data_load import (CenterCrop, FacialKeypointsDataset, Normalize, Rescale,
                       ToTensor)
from models import Net

batch_size = 32
epochs = 10
desired_image_shape = torch.empty(1, 224, 224).size()
desired_keypoints_shape = torch.empty(68, 2).size()
model_dir = './saved_models/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_train_loader(transform):
    train_dataset = FacialKeypointsDataset(
        csv_file='./data/training_frames_keypoints.csv',
        root_dir='./data/training/',
        transform=transform)
    print('Number of train images: ', len(train_dataset))

    assert train_dataset[0]['image'].size(
    ) == desired_image_shape, "Wrong train image dimension"
    assert train_dataset[0]['keypoints'].size(
    ) == desired_keypoints_shape, "Wrong train keypoints dimension"

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader


def initialize_test_loader(transform):
    test_dataset = FacialKeypointsDataset(
        csv_file='./data/test_frames_keypoints.csv',
        root_dir='./data/test/',
        transform=transform)
    print('Number of test images: ', len(test_dataset))

    assert test_dataset[0]['image'].size(
    ) == desired_image_shape, "Wrong test image dimension"
    assert test_dataset[0]['keypoints'].size(
    ) == desired_keypoints_shape, "Wrong test keypoints dimension"

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return test_loader


def evaluate(net, criterion, epoch, test_loader):
    net.eval()
    total_loss = 0
    for batch_i, data in enumerate(test_loader):
        images = data['image']
        keypoints = data['keypoints']
        keypoints = keypoints.view(keypoints.size(0), -1)
        keypoints = keypoints.type(torch.FloatTensor)
        images = images.type(torch.FloatTensor)
        keypoints = keypoints.to(device=device, dtype=torch.float)
        images = images.to(device=device, dtype=torch.float)

        output = net(images)
        loss = criterion(output, keypoints)

        total_loss += loss
    print('Epoch: {}, Test Dataset Loss: {}'.format(
        epoch, total_loss / len(test_loader)))


def train(net, criterion, optimizer, epoch, train_loader, model_id,
          loss_logger):
    # mark as train mode
    net.train()
    # initialize the batch_loss to help us understand
    # the performance of multiple batches
    batches_loss = 0.0
    print("Start training epoch {}".format(epoch))
    for batch_i, data in enumerate(train_loader):
        # a batch of images (batch_size, 1, 224, 224)
        images = data['image']
        # a batch of keypoints (batch_size, 1, 68, 2)
        keypoints = data['keypoints']
        # flatten keypoints
        # from (batch_size, 68, 2) to (batch_size, 136, 1)
        keypoints = keypoints.view(keypoints.size(0), -1)
        # PyTorch likes float type. So we convert to it.
        keypoints = keypoints.to(device=device, dtype=torch.float)
        images = images.to(device=device, dtype=torch.float)

        # forward propagation - calculate the output
        output = net(images)
        # calculate the loss between
        # output keypoints and expected keypoints
        loss = criterion(output, keypoints)

        # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/8
        # https://stackoverflow.com/questions/44732217/why-do-we-need-to-explicitly-call-zero-grad
        # zero the parameter (weight) gradients
        optimizer.zero_grad()

        loss.backward()
        # https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350
        optimizer.step()

        # adjust the running loss
        batches_loss += loss.item()

        if batch_i % 10 == 9:  # print every 10 batches
            print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(
                epoch + 1, batch_i + 1, batches_loss / 10))
            loss_logger.append(batches_loss)
            batches_loss = 0.0


def run():
    print("CUDA is available: {}".format(torch.cuda.is_available()))
    data_transform = transforms.Compose(
        [Rescale(250), CenterCrop(224),
         Normalize(), ToTensor()])

    # loader will split datatests into batches witht size defined by batch_size
    train_loader = initialize_train_loader(data_transform)
    test_loader = initialize_test_loader(data_transform)

    model_id = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    # instantiate the neural network
    net = Net()
    net.to(device=device)
    # define the loss function using SmoothL1Loss
    criterion = nn.SmoothL1Loss()
    # define the params updating function using Adam
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    loss_logger = []

    for i in range(epochs):
        model_name = 'model-{}-epoch-{}.pt'.format(model_id, i)

        # train all data for one epoch
        train(net, criterion, optimizer, i, train_loader, model_id,
              loss_logger)

        # evaludate the accuracy after each epoch
        evaluate(net, criterion, i, test_loader)

        # save model after each epoch of training
        # https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/3
        # https://github.com/pytorch/pytorch/issues/2830
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save({
            'epoch': i,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_logger': loss_logger,
        }, model_dir + model_name)


if __name__ == "__main__":
    run()
