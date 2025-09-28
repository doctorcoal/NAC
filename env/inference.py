import time
import torch
import logging
import torch.backends.cudnn as cudnn
from .data import *
from torch.autograd import Variable
import math

def inference(model, test_loader, args:T_conf, full_dataset=False, beta=0.95):
    '''
    conduct the inference
    1. get and load model according to args.ptf
    2. do the test using the functions in train_new.py
    3. save the model
    '''
    logging.info(f"invoke inference with place layer {model.place_layer}, beta {beta}")
    data_size = args.inf_data_size if not full_dataset else math.inf
    torch.manual_seed(91020)
    model.set_beta(beta)
    args.start_layer = 0
    args.inference = 'hira'
    if 'cuda' in args.type:
        usecuda = True
        cudnn.benchmark = True
    else:
        usecuda = False
        args.gpus = None
    
    model.eval()

    if args.inference == 'exits':
        model.set_eval_pred(args.start_layer, 'exits_forward') 
    elif args.inference == 'normal':
        model.set_eval_pred(args.start_layer, 'normal_forward')
    elif args.inference == 'hira':
        model.set_eval_pred(model.place_layer, 'place_forward')   # place_layer not change (be set the defualt one)

    correct = 0
    total = 0
    if args.inference != 'normal':
        correct_list = [0 for _ in range( model.depth + 1 )]
        total_list = [0 for _ in range( model.depth + 1 )]
    device = torch.device(args.device)
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    model_without_ddp = model.module if args.distributed else model
    model.eval()

    # # warm up
    # print('warm up ...\n')
    # with torch.no_grad():
    #     model.dvfs = 'none' # no dvfs here
    #     for index, data in enumerate( test_loader ): 
    #         if index >= 100: break
    #         images, _ = data
    #         images = Variable(images.type(args.type))
    #         _ = model(images)
    model_without_ddp.initcount()
    if usecuda:
        torch.cuda.synchronize()

    model_time = 0
    model_time_list = []
    exit_dist = []
    logging.debug('start inf')
    st_time = time.perf_counter()
    with torch.no_grad():
        for index, data in enumerate( test_loader ): 
            if index >= data_size: break
            images, labels = data
            images = Variable(images.type(args.type))
            labels = Variable(labels.type(args.type))
            model_st_time = time.perf_counter()
            outputs = model(images)
            model_end_time = time.perf_counter()
            if usecuda:
                torch.cuda.synchronize()
            single_time = model_end_time - model_st_time
            dvfs_time = 0
            if args.inference != 'normal':
                    exit_layer, outputs = outputs
            model_time += single_time
            model_time_list.append(single_time - dvfs_time)  # unlike model_time, here it directly remove the dvfs time
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += ( predicted == labels ).sum().item()
            if args.inference != 'normal':
                total_list[exit_layer] += labels.size( 0 )
                correct_list[exit_layer] += ( predicted == labels ).sum().item()
    end_time = time.perf_counter()
    logging.debug('end inf')
    general_acc = 100 * correct / total
    if args.inference != 'normal':
        acc_list = [correct_list[i]/total_list[i] if total_list[i] != 0 else None for i in range( len( correct_list ) )]
    # logging.info( f'time consumed: {end_time - st_time}' )        

    if args.inference != 'normal':
        for exit_idx in range( len( correct_list ) ):
            if acc_list[exit_idx] != None:
                logging.debug( f'exit{str(exit_idx)}: {100*acc_list[exit_idx]: .3f}% | ')
            else:
                logging.debug( f'exit{str(exit_idx)}: {None} | ')
        logging.debug('')
        model_without_ddp.print_exit_percentage(True)
    time_list = model_without_ddp.output_time()
    count_list = model_without_ddp.output_count()
    exit_layer_ratio = args.exit_layer_ratio if hasattr(args, 'exit_layer_ratio') else 0.136
    nomalized_cost = (count_list[0]+exit_layer_ratio*count_list[1])/total/(args.m_conf['depth'])
    logging.debug( 'total time: %.3f, model time: %.2f (inf: %.2f, exit: %.2f, pred: %.3f, dvfs: %.2f)' %
         (end_time - st_time, model_time, time_list[0], time_list[1], time_list[2], time_list[3]))
    logging.debug( 'inf layer: **%d**, exit count: %d, pred count: %d, nomalized cost: %.2f%%' %
         (count_list[0], count_list[1], count_list[2], nomalized_cost*100))
    if 'E' in args.ptf and 'Q' in args.ptf:
        logging.debug('(optional)nomalized cost: (quantized E) %.4f%% ;(original E) %.3f%%' % 
              (nomalized_cost*100*args.weight*args.activation/32/32,
                (count_list[0]*args.weight*args.activation/32/32+exit_layer_ratio*count_list[1])/total/(args.m_conf['depth'])*100)),
    # logging.info('Accuracy of the network on the {} test images: {:.2f} %%'.format(data_size, general_acc))
    logging.debug('Accuracy of the network on the test images: **{:.2f} %**'.format(general_acc))
    return general_acc, nomalized_cost
