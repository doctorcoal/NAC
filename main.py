from utils import evaluate_policy, str2bool, setup_logging, getdirsize
from datetime import datetime
from DQN import DQN_agent
from env.comp_env import Comp_env
from env.utils import ResultsLog
import utils
import logging
import os, shutil
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import argparse
import torch
import numpy as np
import math

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset for env setup', choices=['cifar10', 'cifar100', 'tiny-imagenet', 'imagenet'])
parser.add_argument('--model', type=str, default='resnet_exit_quant', help='model for env setup', choices=['resnet_exit_quant', 'mobilenetV2_hira_quant', 'DeiT-tiny', 'DeiT-small', 'DeiT-base'])
parser.add_argument('--test', type=int, default=0, help='test setting')
parser.add_argument('--regen_traj', type=str, default='', help='regenerate the past trajectory')
# parser.add_argument('--shallowf', action='store_true', help='shallow finetune mode (different F in comp_env, different save/load)')
parser.add_argument('--arinc', action='store_true', help='ar increase with search steps')
parser.add_argument('--BSD', action='store_true', help='best_score_determine, only needs best score')
parser.add_argument('--suffix', type=str, default='', help='suffix of save file')
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(2e4), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(500), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(60*1000), help='Model evaluating interval, in steps.') 
parser.add_argument('--random_steps', type=int, default=int(300), help='steps for random policy to explore')
parser.add_argument('--noise_decay_interval', type=int, default=int(100), help='')
parser.add_argument('--update_every', type=int, default=15, help='training frequency') # original 50

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=200, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='lenth of sliced trajectory') 
parser.add_argument('--exp_noise', type=float, default=0.25, help='explore noise')
parser.add_argument('--noise_decay', type=float, default=0.99, help='decay rate of explore noise')
parser.add_argument('--Double', type=str2bool, default=True, help='Whether to use Double Q-learning')
parser.add_argument('--Duel', type=str2bool, default=True, help='Whether to use Duel networks')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
logging.info(opt)

def arinc(r_list, step, arinc = False):
    if arinc:
        r_list[1] = r_list[1] * min(0.9*step/2000.0 + 0.1, 1.0)
    return sum(r_list)

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    BriefEnvName = ['comp_search']
    if opt.dataset == 'tiny_imagenet':
        opt.dataset = 'tiny-imagenet'
    env = Comp_env(opt.model, opt.dataset, opt.suffix, opt.test)
    action_dict = dict(zip(env.action_names, range(len(env.action_names))))
    # eval_env = Comp_env(opt.dataset)
    opt.state_dim = env.state_dim
    opt.action_dim = env.action_dim
    opt.max_e_steps = env.tr_step_num
    setup_logging(os.path.join(env.t_conf.save_path, 'log_all.txt'), 
                  os.path.join(env.t_conf.save_path, 'log_info.txt'))

    #Algorithm Setting
    if opt.Duel: algo_name = 'Duel'
    else: algo_name = ''
    if opt.Double: algo_name += 'DDQN'
    else: algo_name += 'DQN'

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info("Random Seed: {}".format(opt.seed))

    logging.info('Algorithm: %s  Env: %s  state_dim: %d  action_dim: %d  Random Seed: %d  max_e_steps: %d',
                algo_name, BriefEnvName[opt.EnvIdex], opt.state_dim, opt.action_dim, opt.seed, opt.max_e_steps)

    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}-{}_S{}_'.format(algo_name,BriefEnvName[opt.EnvIdex],opt.seed) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    # Build model and replay buffer
    if not os.path.exists('model'): os.mkdir('model')
    agent = DQN_agent(**vars(opt))
    results = ResultsLog(os.path.join(env.t_conf.save_path, 'results.csv'), os.path.join(env.t_conf.save_path, 'results_'))
    scores = ResultsLog(os.path.join(env.t_conf.save_path, 'scores.csv'), os.path.join(env.t_conf.save_path, 'scores_'))
    train_scores = ResultsLog(os.path.join(env.t_conf.save_path, 'Tscores.csv'), os.path.join(env.t_conf.save_path, 'Tscores_'))
    if opt.Loadmodel: agent.load(algo_name,BriefEnvName[opt.EnvIdex],opt.ModelIdex)
    if opt.render:
        while True:
            score = evaluate_policy(env, agent, 1)
            logging.info('EnvName: %s  seed: %d  score: %f', BriefEnvName[opt.EnvIdex], opt.seed, score)
    else:
        total_steps = 0
        episodes = 0
        while total_steps < opt.Max_train_steps:
            comp_full = None
            s, info = env.reset(seed=env_seed) # Do not use opt.seed directly, or it can overfit to opt.seed
            env_seed += 1
            done = False
            total_score = 0
            total_ar = 0
            total_bcr = 0
            total_mcr = 0
            best_score = 0
            best_bcrs = 0
            best_mcrs = 0
            best_crs = 0
            best_as = 0
            best_step = 0

            '''Interact & train'''
            while not done:
                #e-greedy exploration
                if opt.regen_traj:
                    mem_limit = 1.0
                    if total_steps >= len(opt.regen_traj):
                        exit()
                    a = action_dict[opt.regen_traj[total_steps]]
                else:
                    if total_steps < opt.random_steps:
                        mem_limit = 0.1
                        if np.random.rand() < 0.6:  # 50% percent select default finetune
                            a = 0
                        else:
                            a = env.random_action()
                    else: 
                        mem_limit = 1.0
                        a = agent.select_action(s, deterministic=False, mask=comp_full)
                logging.info("###### entering Step "+str(total_steps))
                logging.info('action: '+env.action_names[a])
                
                # dvc = next(env.model.parameters()).device
                # logging.info(dvc)
                # if dvc != torch.device("cuda:0"):
                #     logging.info("wrong device")
                #     logging.info(dvc)
                #     exit()
                    
                envfilename = os.path.join(env.t_conf.env_path, env.trajectory+env.action_names[a])
                if os.path.isfile(envfilename):   # trajectory memory hit
                    logging.info("# trajectory hit")
                    newstep = torch.load(envfilename)
                    s_next, r, dw, tred, info, comp_full = newstep['output']
                    envpasttime = env.time
                    envweight = env.r_weight
                    envtconf = env.t_conf
                    del env
                    env = newstep['env'] # avoid load previous env's expired time
                    env.r_weight = envweight
                    env.time = envpasttime
                    env.t_conf = envtconf
                    info = env.weighted_r() + [0.]
                    r = sum(info)  # ignore bonus
                else:
                    logging.info("# trajectory add new step {}".format(env.action_names[a]))
                    s_next, _, dw, tred, _, comp_full, valid_action = env.step(a) # dw: dead&win; tred: truncated
                    info = env.weighted_r() + [0.]
                    r = sum(info)  # ignore bonus
                    if valid_action: # prevent invalid trajectory saved (as envfilename dont judge valid or not)
                        mem_size = getdirsize(env.t_conf.env_path)
                        if mem_size < mem_limit*1500*1000000000:
                            logging.info("# trajectory save (current size: {:.2f}G)".format(mem_size/1024**3))
                            utils.save_on_master({'output': [s_next, r, dw, tred, info, comp_full], 'env': env}, os.path.join(env.t_conf.env_path, env.trajectory))
                logging.info('current trajectory: '+env.trajectory)
                logging.info('current full trajectory: '+env.full_trajectory)
                
                total_score += r
                total_ar += info[1]
                total_bcr += info[3]
                total_mcr += info[4]
                if total_score > best_score:
                    if opt.BSD:
                        r = total_score - best_score
                    best_step = total_steps
                    best_score = total_score
                    best_bcrs = total_bcr
                    best_mcrs = total_mcr
                    best_crs = best_bcrs + best_mcrs
                    best_as = total_ar
                else:
                    if opt.BSD:
                        r = 0
                    
                logging.info("###### s_next "+str(s_next))
                logging.info("###### s_name "+str(env.state_names))
                logging.info("###### r "+str(r))
                logging.info("########### dr "+str(info[0]))
                logging.info("########### ar "+str(info[1]))
                logging.info("########### tr "+str(info[2]))
                logging.info("########### br "+str(info[3]))
                logging.info("########### mr "+str(info[4]))
                logging.info("########### bonus "+str(info[5]))
                logging.info("############### total score "+str(total_score))
                logging.info("########### total_as "+str(total_ar))
                logging.info("########### total_bcs "+str(total_bcr))
                logging.info("########### total_mcs "+str(total_mcr))
                logging.info("########### total_crs "+str(total_bcr+total_mcr))
                logging.info("############### best score "+str(best_score))
                logging.info("########### best bcrs "+str(best_bcrs))
                logging.info("########### best mcrs "+str(best_mcrs))
                logging.info("########### best crs "+str(best_crs))
                logging.info("########### best as "+str(best_as))
                logging.info("!!!!!!Done!!!!!!!" if dw else '')
                logging.info("!!!!!!truncated!!!!!!" if tred else '')
                logging.info('\n\n')
                done = (dw or tred)
                results.add(step=total_steps, acc=env.acc, bcr=env.bcr_total(), mcr=env.mcr, r=r, 
                            dr = info[0], ar = info[1], tr = info[2], br = info[3], mr = info[4], bonus=info[5], 
                            total_ar = total_ar, total_bcr = total_bcr, total_mcr = total_mcr)
                if done:
                    episodes += 1
                    logging.info("########### total score"+str(total_score) if dw else '')
                    train_scores.add(episode = episodes, step=total_steps, score=total_score, best_as=best_as, best_step=best_step,
                                     best_bcrs=best_bcrs, best_mcrs=best_mcrs, best_crs=best_crs ,best_score=best_score)
                    train_scores.save()
                
                if opt.BSD:
                    decayed_r = r
                else:
                    decayed_r = arinc(info, total_steps, opt.arinc)
                agent.replay_buffer.add(s, a, decayed_r, s_next, dw)
                s = s_next

                '''Update'''
                # train n times every n steps rather than 1 training per step. Better!
                if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
                    logging.info('train agent '+str(opt.update_every) +' times.')
                    for j in range(opt.update_every): agent.train()

                '''Noise decay & Record & Log'''
                if total_steps % opt.noise_decay_interval == 0: agent.exp_noise *= opt.noise_decay
                total_steps += 1
                # if total_steps % opt.eval_interval == 0:
                #     logging.info("eval DQN model")
                #     score = evaluate_policy(eval_env, agent, turns = 1)
                #     scores.add(step=total_steps, score=score)
                #     if total_steps / opt.eval_interval >= 5:
                #         scores.plot('step', 'score', '', 'step', 'score')
                #         scores.save()
                #     # if opt.write:
                #     #     writer.add_scalar('ep_r', score, global_step=total_steps)
                #     #     writer.add_scalar('noise', agent.exp_noise, global_step=total_steps)
                #     logging.info("eval finish: step {}, score {:.2f}".format(total_steps, score))

                if total_steps % 10 == 0:
                    logging.info('ploting...')
                    if not train_scores.empty: # first done happens
                        train_scores.plot('episode', 'score', '', 'episode', 'total score', '_epi')
                        train_scores.plot('step', 'score', '', 'step', 'total score', '_step')
                        train_scores.plot('episode', 'best_score', '', 'episode', 'best score', '_bestepi')
                        train_scores.plot('step', 'best_score', '', 'step', 'best score', '_beststep')
                        
                        train_scores.plot('step', 'best_as', '', 'step', 'best as', '_beststep')
                        train_scores.plot('step', 'best_bcrs', '', 'step', 'best bcrs', '_beststep')
                        train_scores.plot('step', 'best_mcrs', '', 'step', 'best mcrs', '_beststep')
                        train_scores.plot('step', 'best_crs', '', 'step', 'best crs', '_beststep')
                    # results.plot('step', 'r', '', 'step', 'reward')
                    # results.plot('step', 'acc', '', 'step', 'accuracy')
                    # results.plot('step', 'bcr', '', 'step', 'bitops compression ratio')
                    # results.plot('step', 'mcr', '', 'step', 'memory compression ratio')
                    # results.plot('step', 'dr', '', 'step', 'accuracy drop reward')
                    # results.plot('step', 'ar', '', 'step', 'accuracy change reward')
                    # results.plot('step', 'tr', '', 'step', 'time reward')
                    # results.plot('step', 'br', '', 'step', 'bitops CR reward')
                    # results.plot('step', 'mr', '', 'step', 'memory CR reward')
                    # results.plot('step', 'bonus', '', 'step', 'bonus reward')
                    # results.plot_m('step', ['r', 'dr', 'ar', 'tr', 'br', 'mr', 'bonus'], 
                    #                 ['total', 'accuracy drop', 'accuracy change', 'time', 'bitops CR', 'memory CR', 'bonus'],
                    #                 'with total', 'step', 'rewards')
                    # results.plot_m('step', ['dr', 'ar', 'tr', 'br', 'mr', 'bonus'],
                    #                 ['accuracy drop', 'accuracy change', 'time', 'bitops CR', 'memory CR', 'bonus'],
                    #                 '', 'step', 'rewards')
                    results.save()
                    logging.info('finish.')
                    
                # '''save model'''
                # if total_steps % opt.save_interval == 0:
                #     logging.info('save DQN model')
                #     agent.save(algo_name,BriefEnvName[opt.EnvIdex],int(total_steps/100),os.path.join('DQN_save',opt.dataset,env.t_conf.name))
    # env.close()
    # eval_env.close()

if __name__ == '__main__':
    main()








