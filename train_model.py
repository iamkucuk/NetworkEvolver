import copy
import os
import sys
import time
import torch
# from livelossplot import PlotLosses
from torch.utils.tensorboard import SummaryWriter


def train_model(model_name, model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=5, scheduler=None,
                device=None):
    if not os.path.exists('models/' + str(model_name)):
        os.makedirs('models/' + str(model_name))
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter("runs/" + model_name)
    writer.add_graph(model, torch.zeros([1, 3, 64, 64]).to(device))
    writer.close()
    since = time.time()
    best_acc = 0.0
    best = 0
    beggining_loss = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                print("\rIteration: {}/{}, Loss: {}.".format(i + 1, len(dataloaders[phase]),
                                                             loss.item() * inputs.size(0)), end="")

                if i % 1000 == 999 and phase == 'train':
                    writer.add_scalar('training loss',
                                      running_loss / 1000,
                                      epoch * len(dataloaders['train']) + i)

                if epoch == 1 and i == 100:
                    beggining_loss = loss.item() * inputs.size(0)
                if epoch == 1 and i == 5000:
                    if loss.item() * inputs.size(0) > beggining_loss - 0.01:
                        return 10000
                #                 print( (i+1)*100. / len(dataloaders[phase]), "% Complete" )
                sys.stdout.flush()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]



            if phase == 'train':
                avg_loss = epoch_loss
                t_acc = epoch_acc
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc

            #             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #                 phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc

                best = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())

        print('Train Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, t_acc))
        print('Val Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))
        print()
        torch.save(model.state_dict(), './models/' + str(model_name) + '/model_{}_epoch.pt'.format(epoch + 1))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Accuracy: {}, Epoch: {}'.format(best_acc, best))

    return val_loss
