import os
import argparse
import torch

from tensorboardX import SummaryWriter

from models.srresnet import SRResNet
from models.edsr import EDSR
from models.wdsr import WDSR_A
from dataset import CityScapesDataset
from torchvision.utils import save_image

"""
CityScapes segmentation class labels:

0  - unlabeled        1  - ego vehicle      2  - rectification border  3  - out of roi
4  - static           5  - dynamic          6  - ground

7  - road             8  - sidewalk         9  - parking            10 - rail track

11 - building         12 - wall             13 - fence              14 - guard rail
15 - bridge           16 - tunnel

17 - pole             18 - polegroup        19 - traffic light      20 - traffic sign
21 - vegetation       22 - terrain          23 - sky

24 - person           25 - rider

26 - car              27 - truck            28 - bus                29 - caravan
30 - trailer          31 - train            32 - motorcucle         33 - bicycle
"""

parser = argparse.ArgumentParser(description="EquiSurf Super-resolution Experiments")
parser.add_argument('--name', type=str, help='Set experiment name', required=True)
parser.add_argument('--model', type=str, help='Set SR Model to use', required=True)
parser.add_argument('--scale', type=int, help='Set super-resolution factor', required=True)
parser.add_argument('--crop_size', type=int, default=None, help='Set desired image dimension (crop_size x crop_size)')

"""Example Usage: ... --train_lbls 26 27 28 29 ... """
parser.add_argument('--train_lbls', nargs='+', default=None, help='Set labels to be used in training => default: full image is used')
parser.add_argument('--test_lbls', nargs='+', default=None, help='Set labels to be used in testing => default: full image is used')
parser.add_argument('--pretrain', type=str, default=None, help='Set path to pretrained weights => default: no pretrained weights')

parser.add_argument('--gpu', type=str, default='0', help='Set gpu id to use => default: 0')
parser.add_argument('--batch_size', type=int, default=16, help='Set batch size during training => default: 16')
parser.add_argument('--val_max', type=int, default=None, help='Set max size for validation set => default: len(val_dataset)')
parser.add_argument('--save_epochs', type=int, default=1, help='Set checkpoint save interval => default: 1')
parser.add_argument('--val_epochs', type=int, default=1, help='Set validation interval => default: 1')
parser.add_argument('--log_iters', type=int, default=100, help='Set training log interval (number of images) => default: 100')


class Experiment():
    def __init__(self, name, model, scale, pretrain):

        if model == 'srresnet':
            self.model = SRResNet(scale)
            self.lr = 1e-4
            self.num_epochs = 500
            self.lr_update_epochs = None
            self.lr_update_factor = None
            self.loss = torch.nn.MSELoss(reduction='sum')
            print("Created model SRResNet")

        elif model == 'edsr':
            self.model = EDSR(scale)
            self.lr = 1e-4
            self.num_epochs = 500
            self.lr_update_epochs = 50
            self.lr_update_factor = 0.5
            self.loss = torch.nn.L1Loss(reduction='sum')
            print("Created model EDSR")

        elif model == 'wdsr':
            self.model = WDSR_A(scale)
            self.lr = 1e-3
            self.num_epochs = 500
            self.lr_update_epochs = 50
            self.lr_update_factor = 0.5
            self.loss = torch.nn.L1Loss(reduction='sum')
            print("Created model WDSR")

        tensorboard_dir = os.path.join('tensorboard', name)
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        self.writer = SummaryWriter(tensorboard_dir)

        self.checkpoints_dir = os.path.join('checkpoints', name)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        if not pretrain is None:
            print("===> Loaded model with weights from: " + pretrain)
            self.model.load_state_dict(torch.load(pretrain))

        self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


def main():
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if not torch.cuda.is_available():
        raise Exception('Selected GPU with id {} not found'.format(args.gpu))

    train_dataset = CityScapesDataset(args.train_lbls, args.scale, 'train', crop_size=args.crop_size)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=args.batch_size, shuffle=True)
    print("Loaded training dataset with {} images and labels:".format(len(train_dataset)))
    print(args.train_lbls)

    val_dataset = CityScapesDataset(args.test_lbls, args.scale, 'val', crop_size=args.crop_size, maximum=args.val_max)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, num_workers=1, batch_size=1)
    print("Loaded val dataset with {} images and labels:".format(len(val_dataset)))
    print(args.test_lbls)

    experiment = Experiment(args.name, args.model, args.scale, args.pretrain)

    for epoch in range(1, experiment.num_epochs + 1):
        update_learning_rate(experiment, epoch)
        train(experiment, train_dataloader, epoch, args.log_iters)

        if epoch % args.save_epochs == 0:
            save(experiment.model, os.path.join(experiment.checkpoints_dir, 'model_epoch_{}.pth'.format(epoch)))

        if epoch % args.val_epochs == 0:
            with torch.no_grad():
                validate(experiment.model, experiment.loss, val_dataloader, experiment.writer, epoch)

def update_learning_rate(experiment, epoch):
    if experiment.lr_update_epochs and epoch % experiment.lr_update_epochs == 0:
        print('\nUpdate learning rate by factor: {}'.format(experiment.lr_update_factor))
        experiment.lr *= experiment.lr_update_factor
        for p_group in experiment.optimizer.param_groups:
            p_group['lr'] = experiment.lr

def save(model, path):
    print('\nCheckpoint saved to ' + path)
    torch.save(model.state_dict(), path)

def train(experiment, dataloader, epoch, log_iters):
    print('\nEpoch={}, lr={}'.format(epoch, experiment.optimizer.param_groups[0]['lr']))
    experiment.model.train()

    running_loss = 0.0
    images_computed = 0

    for iteration, batch in enumerate(dataloader, 1):
        overall_iter = iteration * dataloader.batch_size + (epoch-1) * len(dataloader.dataset)

        if len(batch) == 3:
            mask = batch[2]
            num_pixels = torch.sum(mask == 1).data.item()
            if num_pixels == 0:
                print("===> Epoch[{}]({}/{}): no pixels in batch correspond to specified labels".format(epoch, iteration, len(dataloader)))
                del mask
                continue
            mask = mask.cuda()

        input, gt = batch[0].requires_grad_(True).cuda(), batch[1].cuda()
        output = experiment.model(input)

        if len(batch) == 3:
            output = output * mask
            gt = gt * mask
        else:
            num_pixels = output.numel()

        loss = experiment.loss(output, gt) / num_pixels
        experiment.optimizer.zero_grad()
        loss.backward()
        experiment.optimizer.step()

        running_loss += loss.data.item()
        images_computed += dataloader.batch_size

        if recorded_datapoints >= log_iters:
            # computes the avg per pixel loss over current interval
            data = running_loss / (images_computed / dataloader.batch_size)
            running_loss = 0.0
            images_computed = 0
            print("===> Epoch[{}]({}/{}): Loss: {:.5}".format(epoch, iteration, len(dataloader), data))
            experiment.writer.add_scalar("Train/Loss", data, overall_iter)

def validate(model, criterion, dataloader, writer, epoch):
    dataset_size = len(dataloader.dataset)
    print('\nValidation: Epoch={}, Dataset_Size={}'.format(epoch, dataset_size))
    model.eval()

    avg_loss = 0.0
    skipped_samples = 0
    for iteration, sample in enumerate(dataloader):

        if len(sample) == 3:
            mask = sample[2].cuda()
            num_pixels = torch.sum(mask == 1).data.item()
            if num_pixels == 0:
                skipped_samples += 1
                del mask
                continue

        input, gt = sample[0].cuda(), sample[1].cuda()
        output = model(input)

        if len(sample) == 3:
            output = output * mask
            gt = gt * mask
        else:
            num_pixels = output.numel()

        avg_loss += criterion(output, gt).data.item() / num_pixels

        output = output * 0.5 + 0.5
        save_image(output, 'test.png')


    avg_loss /= (dataset_size - skipped_samples)
    print("===> Loss: {:.5}".format(avg_loss))
    writer.add_scalar('Validation/Loss', avg_loss, epoch)


if __name__ == '__main__':
    main()
