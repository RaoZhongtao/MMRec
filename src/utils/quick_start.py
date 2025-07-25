# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os
import wandb

def initWandb(model, dataset, extractor, config, hyper_idx, moe_num):
    wandb_run_name = f"{model}_{dataset}_{extractor}_moe{moe_num}_{hyper_idx}"
    wandb_config = {
        "project": "MMRec", 
        "entity": "MultimodelRec",
        "config": config,
        "name": wandb_run_name
    }
    
    wandb.init(**wandb_config)

def quick_start(model, dataset, config_dict, ckpt_dir, save_model=True, mg=False, mode='train', extractor='default', moe_num=0):
    # merge config dict
    best_model_path = f"../checkpoints/{model}_{dataset}_best.pth"
    config = Config(model, dataset, config_dict, mg, extractor, moe_num)
    

    
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    recDataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(recDataset))

    train_dataset, valid_dataset, test_dataset = recDataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, mode, batch_size=config['train_batch_size'], shuffle=False)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0
    best_valid_score = -1
    logger.info('\n\n=================================\n\n')

    if mode=='train':
        # hyper-parameters
        hyper_ls = []
        if "seed" not in config['hyper_parameters']:
            config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
        for i in config['hyper_parameters']:
            hyper_ls.append(config[i] or [None])
        # combinations
        combinators = list(product(*hyper_ls))
        total_loops = len(combinators)
        for hyper_tuple in combinators:
            
            initWandb(model, dataset, extractor, config, idx, moe_num)
            
            # random seed reset
            for j, k in zip(config['hyper_parameters'], hyper_tuple):
                config[j] = k
            init_seed(config['seed'])

            logger.info('========={}/{}: Parameters:{}={}======='.format(
                idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

            # set random state of dataloader
            train_data.pretrain_setup()
            # model loading and initialization
            model = get_model(config['model'])(config, train_data).to(config['device'])
            logger.info(model)

            # trainer loading and initialization
            trainer = get_trainer()(config, model, best_valid_score, mg)
            # debug
            
            # model training
            best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, best_model_path, valid_data=valid_data, test_data=test_data, saved=save_model)
            hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

            # save best test
            if best_test_upon_valid[val_metric] > best_test_value:
                best_test_value = best_test_upon_valid[val_metric]
                best_test_idx = idx
            

            logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
            logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
            logger.info('████Current BEST████:\nParameters: {}={},\n'
                        'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
                hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))
            
            print(f"best_test_upon_valid: {best_test_upon_valid}")
            try:
                wandb.log({
                    "H@10": best_test_upon_valid["recall@10"],
                    "H@20": best_test_upon_valid["recall@20"],
                    "N@10": best_test_upon_valid["ndcg@10"],
                    "N@20": best_test_upon_valid["ndcg@20"],
                    "Tail H@10": best_test_upon_valid["Tail HR@10"],
                    "Tail H@20": best_test_upon_valid["Tail HR@20"],
                    "Tail N@10": best_test_upon_valid["Tail NDCG@10"],
                    "Tail N@20": best_test_upon_valid["Tail NDCG@20"],
                })
            except Exception as e:
                logger.error(f"KeyError: {e} not found in best_test_upon_valid: {best_test_upon_valid}")
                
            idx += 1
            #########
    else:
        model = get_model(config['model'])(config, train_data).to(config['device'])
        test = True
        trainer = get_trainer()(config, model, best_valid_score=-1, mg=mg, test=test)
        trainer.load_model(ckpt_dir)
        test_result = trainer.evaluate(test_data, True)
        logger.info('test result: {}'.format(dict2str(test_result)))

    # log info
    logger.info('\n============All Over=====================')
    if mode == 'train':
        for (p, k, v) in hyper_ret:
            logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                    p, dict2str(k), dict2str(v)))

        logger.info('\n\n█████████████ BEST ████████████████')
        logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   dict2str(hyper_ret[best_test_idx][1]),
                                                                   dict2str(hyper_ret[best_test_idx][2])))
    else:
        for (params, test) in hyper_ret:
            logger.info('Parameters: {}={},\n Test: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                    params, dict2str(test)))
        logger.info('\n\n█████████████ Test ████████████████')

