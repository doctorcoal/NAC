import time
import logging
import torch
from .data import T_conf
from torch.autograd import Variable
from .utils import AverageMeter, adjust_learning_rate, accuracy

def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None, onlyexit=False, args:T_conf=None, place = True, amp=True):
    place = True if (place and len(model.place_layer) > 0) else False
    model.set_train('place' if place else 'original', onlyexit = onlyexit)
    device = torch.device(args.device)
    model.to(device)
    if training:
        model.train() # onlyexit means only train exit layers and not require last output
    else:
        model.eval()
    if place:
        output_len = len(model.place_layer) + (0 if onlyexit else 1)
        place_layer = sorted(model.place_layer + ([] if onlyexit else [model.exit_num]))
        top1 = [AverageMeter() for i in range(output_len)]
        top5 = [AverageMeter() for i in range(output_len)]
    else:
        top1 = AverageMeter()
        top5 = AverageMeter()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_exit = AverageMeter()
    losses_last = AverageMeter()

    end = time.time()
    for i, (inputs, target) in enumerate(data_loader):
        # if training: 
        #     if args.lr_decay == 'cos':
        #         adjust_learning_rate(optimizers, epoch, i, len(data_loader), args)
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            target = target.cuda()

        if not training:
            with torch.no_grad():
                input_var = Variable(inputs.type(args.type))
                target_var = Variable(target)
                # compute output
                if amp:
                    with torch.cuda.amp.autocast():
                        output = model(input_var)
                else:
                    output = model(input_var)
        else:
            input_var = Variable(inputs.type(args.type))
            target_var = Variable(target)
            # compute output
            if amp:
                with torch.cuda.amp.autocast():
                    output = model(input_var)
            else:
                output = model(input_var)
        if place:
            loss_exit = 0.0
            for outputdata in output[:-1]:
                loss_exit += criterion(outputdata, target_var)
            loss_last = criterion(output[-1], target_var)
            loss = loss_exit+loss_last
        else:
            loss = criterion(output, target_var)
        if torch.isnan(loss):
            logging.warning('######INF LOSS OCCURED!######')

        if training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            if loss:
                loss.backward()
            optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        if place:
            if not onlyexit:
                losses_last.update(loss_last.item(), inputs.size(0))
            else:
                loss_exit = loss
            losses_exit.update(loss_exit.item(), inputs.size(0))
            # measure accuracy and record loss
            for j, data in enumerate(output):
                prec1, prec5 = accuracy(data, target, topk=(1, 5))
                top1[j].update(prec1.item(), inputs.size(0))
                top5[j].update(prec5.item(), inputs.size(0))
        else:
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))
        torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print_freq = args.print_freq/5 if not training and args.dataset == 'imagenet' else args.print_freq
        if place:
            if i % (print_freq * 5) == 0:
                logging.debug('{phase} - Epoch: [{0}][{1}/{2}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                                    epoch, i, len(data_loader),
                                    phase='TRAINING' if training else 'EVALUATING',
                                    batch_time=batch_time, data_time=data_time
                                ))
                for j in range(output_len):
                    logging.debug('exit layer {layer}\t'
                                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                    layer = place_layer[j], top1=top1[j], top5=top5[j]))
                logging.debug('exit loss {loss.val:.3f} ({loss.avg:.3f})'.format(loss=losses_exit))
                if not onlyexit:
                    logging.debug('last loss {loss.val:.3f} ({loss.avg:.3f})'.format(loss=losses_last))
        else:
            if i % print_freq == 0:
                logging.debug('{phase} - Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                epoch, i, len(data_loader),
                                phase='TRAINING' if training else 'EVALUATING',
                                batch_time=batch_time,
                                data_time=data_time, loss=losses, top1=top1, top5=top5))
    if place:
        for j in range(output_len):
            logging.debug('['+('Train' if training else 'Test')+' end] exit layer {layer}\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            layer = 'last' if place_layer[j] == place_layer[-1] else place_layer[j], top1=top1[j], top5=top5[j]))
        return sum([i.avg for i in top1]), ((losses_last.avg, losses_exit.avg) if not onlyexit else losses_exit.avg)
    else:
        logging.debug('['+('Train' if training else 'Test')+' end] Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))
        return top1.avg, losses.avg