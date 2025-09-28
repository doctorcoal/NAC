import torch
import torch_pruning as tp
# from nikolaos.bof_utils import LogisticConvBoF
# from nikolaos.bof_utils_float import LogisticConvBoF as Bof_q
from models.binarymodule import IRlinear
from models.vit_pytorch.distill_comp import Deit_Comp
from models.vit_pytorch.vit import Attention
from models.vit_pytorch.exit_modules import GlobalSparseAttn
from functools import partial
import logging

def progressive_pruning(pruner: tp.pruner.GroupNormPruner, model, speed_up, example_inputs):
    model.eval()
    if type(model) is Deit_Comp:
        base_ops, base_para = model.update_flops_params()
    else:
        base_ops, base_para = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    current_speed_up = 1
    while current_speed_up < speed_up:
        pruner.step(interactive=False)
        if type(model) is Deit_Comp:
            head_id = 0
            for m in model.modules():
                if isinstance(m, Attention):
                    # print("Head #%d"%head_id)
                    # print("[Before Pruning] Num Heads: %d, Head Dim: %d =>"%(m.heads, m.head_dim))
                    m.heads  = pruner.num_heads[m.to_qkv]
                    m.head_dim = m.to_qkv.out_features // (3 * m.heads)
                    # print("[After Pruning] Num Heads: %d, Head Dim: %d"%(m.heads, m.head_dim))
                    # print()
                    head_id+=1
                # elif isinstance(m, GlobalSparseAttn):
                #     m.num_heads  = pruner.num_heads[m.qkv]
                #     m.head_dim = m.qkv.out_features // (3 * m.num_heads)
            pruned_ops, pruned_para = model.update_flops_params()
        else:
            pruned_ops, pruned_para = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        current_speed_up = float(base_ops) / pruned_ops
        if pruner.current_step == pruner.iterative_steps:
            break
        #logging.info(current_speed_up)
    return current_speed_up, float(base_para)/pruned_para

def pruning(model, example_inputs, pruning_ratio=0.1):
    # model.BNnoquant(True)
    # 1. Importance criterion
    ignored_layers = []
    unwrapped_parameters = []
    if type(model) is Deit_Comp:
        train_mode = "place"
        unwrapped_parameters = []
        num_heads = {}
        for m in model.modules():
            if isinstance(m, Attention):
                num_heads[m.to_qkv] = m.heads 
            # if isinstance(m, GlobalSparseAttn):
            #     num_heads[m.qkv] = m.num_heads 
        imp = tp.importance.GroupNormImportance(p=2, normalizer='max') # normalized by the maximum score for CIFAR
        pruner_entry = partial(tp.pruner.GroupNormPruner, num_heads=num_heads, reg=5e-4, global_pruning=True, isomorphic=True)
    else:
        train_mode = "place"
        imp = tp.importance.GroupNormImportance(p=2, normalizer='max') # normalized by the maximum score for CIFAR
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=5e-4, global_pruning=True, isomorphic=True)
    # imp = tp.importance.MagnitudeImportance(p=1) # normalized by the maximum score for CIFAR
    # pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=False)

    # train_mode = "original" # currently forbidden prune the fcs in exit layer
    model.set_train(train_mode)
    
    current_place = model.place_layer # backup place_layer
    model.place_layer = model.init_place_layer
    # model.place_layer = list(range(model.conv_in_singlelayer-1, # ensure all exit layer recognized by pruner
    #                                 model.depth,model.conv_in_singlelayer))
    # 2. Initialize a pruner with the model and the importance criterion
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, IRlinear)) and (m.out_features == model.num_classes) or (type(model.last) is not torch.nn.Identity and m in model.last):
            ignored_layers.append(m) # DO NOT prune the final classifier!
        elif train_mode == "original":
            if m in model.exit_list:
                 ignored_layers.append(m) # ignore all exit layers if original
        elif isinstance(m, GlobalSparseAttn): 
            ignored_layers.append(m)
    logging.debug("ignored_layers:")
    logging.debug(ignored_layers)
    pruning_ratio_dict = {}
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=1,
        pruning_ratio=pruning_ratio,
        # pruning_ratio_dict=pruning_ratio_dict,
        max_pruning_ratio=1.0,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
    )

    # 4. Prune & finetune the model
    # base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    logging.debug("MODEL before prune:")
    logging.debug(model)
    bitops_cr, memory_cr = progressive_pruning(pruner, model, speed_up=1.0/(1-0.9), example_inputs=example_inputs)
    logging.debug("MODEL after prune:")
    logging.debug(model)
    del pruner
    # macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    # logging.info("ratio"+str(pruning_ratio)+" ops before prunig: "+str(base_macs)+" params before prunig: "+str(base_nparams))
    # logging.info("ops after prunig: "+str(macs)+" params after prunig: "+str(nparams))
    model.zero_grad() # Remove gradients
    model.place_layer = current_place
    return bitops_cr, memory_cr