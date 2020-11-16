from dataset import Cifar100Dataset
from model import *
from metrics.metrics import accuracy
from augmentations.augmentations import TrainAugment, TestAugment
import numpy as np


# TODO:
# stochastic depth (Huang et al., 2016) with drop connect ratio 0.3.


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def main():

    torch.manual_seed(2019)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2019)

    device = 'cuda:0'
    ###################################
    ### Dataset
    ###################################
    train_dataset = Cifar100Dataset(root='./input/', train=True, download=True, transform=TrainAugment())
    test_dataset = Cifar100Dataset(root='./input/', train=False, download=True, transform=TestAugment())

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=80,
                                                   shuffle=True, num_workers=8,
                                                   pin_memory=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=80,
                                                  shuffle=False, num_workers=8,
                                                  pin_memory=False)

    #########################################
    # model
    model = efficientnet_b7(num_classes=100).to(device)

    # print(model)

    # Optimizer
    # torch.optim : https://pytorch.org/docs/stable/optim.html
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.016, momentum=0.9, weight_decay=0.9)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # Scheduler
    # torch.optim.lr_scheduler : https://pytorch.org/docs/stable/optim.html?highlight=lr_scheduler#torch.optim.lr_scheduler.LambdaLR
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(2.4*len(train_dataloader)), gamma=0.97)
    # Loss and Evalaiton function
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    # loss_fn = torch.nn.MSELoss().to(device)
    eval_fn = accuracy

    epoch = 500

    for i in range(epoch):
        train_loss, train_accuracy = train(model, optimizer, train_dataloader,
                                               device, loss_fn, eval_fn, i, scheduler)
        test_loss, test_accuracy = test(model, test_dataloader,
                                            device, loss_fn, eval_fn)
        print(f'''--------   epoch {i:>3} (lr: {scheduler.get_lr()[0]:.5f})  --------
train loss = {train_loss:.5f} | train acc = {train_accuracy:.2%} |
test loss = {test_loss:.5f} | test acc = {test_accuracy:.2%}''')

    print('=== Success ===')


def l2_loss(model):
    loss = 0.0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            for p in m.parameters():
                loss += (p ** 2).sum() / 2 #p.norm(2)

    return loss


def train(model, optimizer, dataloader, device, loss_fn, eval_fn, epoch, scheduler=None):
    model.train()
    avg_loss = 0
    avg_accuracy = 0

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    for step, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        preds = logits.softmax(dim=1)
        loss = loss_fn(logits, targets.argmax(dim=1)) 
        loss += 1e-5 * l2_loss(model)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        avg_accuracy += eval_fn(preds, targets)
        
        # print(preds)
        # print(targets)
        # print(preds.shape)
        # print(targets.shape)
        acc1, acc5 = accuracy_1(preds, targets, device, topk=(1, 5))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        if scheduler is not None:
            scheduler.step()

    avg_loss /= len(dataloader)
    avg_accuracy /= len(dataloader)
    return avg_loss, avg_accuracy


def test(model, dataloader, device, loss_fn, eval_fn):
    # model.eval()
    # avg_loss = 0
    # avg_accuracy = 0
    # with torch.no_grad():
    #     for inputs, targets in dataloader:
    #         inputs = inputs.to(device)
    #         targets = targets.to(device)
    #         logits = model(inputs)
    #         preds = logits.softmax(dim=1)
    #         loss = loss_fn(logits, targets.argmax(dim=1)) 
    #         loss += 1e-5 * l2_loss(model)
    #         avg_loss += loss.item()
    #         avg_accuracy += eval_fn(preds, targets)

    # avg_loss /= len(dataloader)
    # avg_accuracy /= len(dataloader)
    # return avg_loss, avg_accuracy

    # correct_1 = 0.0
    # correct_5 = 0.0

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.eval()
    avg_loss = 0
    avg_accuracy = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)

            preds = logits.softmax(dim=1)
            loss = loss_fn(logits, targets.argmax(dim=1)) 
            loss += 1e-5 * l2_loss(model)
            avg_loss += loss.item()
            avg_accuracy += eval_fn(preds, targets)

            acc1, acc5 = accuracy_1(preds, targets, device, topk=(1, 5))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            # _, pred = logits.topk(5, 1, largest=True, sorted=True)
            # targets = targets.view(targets.size(0), -1).expand_as(pred)
            # correct = pred.eq(targets).float()

            # #compute top 5
            # correct_5 += correct[:, :5].sum()

            # #compute top1 
            # correct_1 += correct[:, :1].sum()

            # print("Top 1 err: ", 1 - correct_1 / len(dataloader.dataset))
            # print("Top 5 err: ", 1 - correct_5 / len(dataloader.dataset))
        
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    
    # avg_accuracy = top1.avg

    avg_loss /= len(dataloader)
    avg_accuracy /= len(dataloader)

    return avg_loss, avg_accuracy


def accuracy_1(output, target, device, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, _target = target.max(-1)

        # print(_target)

        # print(type(_target))

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(_target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
