import numpy as np
from utils import *
import matplotlib.pyplot as plt
import argparse
from dataloader import load_MNIST, load_minibatch
from model import Rookie
from optimizer import opt_SGD
from train import trainer, test
import os
import pickle


def main(args):
    log_dir = args.logpath + '{}_{}_{}/'.format(args.date,args.learning_rate, args.batch_size)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    (x_train,y_train,x_val,y_val,x_test,y_test) = load_MNIST(dataroot=args.datapath, rand_seed=args.seed)
    mini_batch_train = load_minibatch(x_train,y_train,batch_size=args.batch_size,seed=args.seed)
    mini_batch_valid = load_minibatch(x_val,y_val,batch_size=args.batch_size,seed=args.seed)
    mini_batch_test = load_minibatch(x_test,y_test,batch_size=args.batch_size,seed=args.seed)
    
    model = Rookie(in_channel=784,hidden_layer=[500,300],num_classes=10)
    opt = opt_SGD(parameters=model.parameters,learning_rate=args.learning_rate)
    
    Trainer = trainer(model, opt,args.epoch, mini_batch_train, mini_batch_valid, log_dir)
    Trainer.train()
    if args.plot:
        plot_metrics(Trainer, log_dir)
    
    with open(log_dir + 'best_model.pkl', 'rb') as f:
        checkpoint = pickle.load(f)
    model.load_parameters(checkpoint['parameters'])
    test(model,mini_batch_test)
    
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default= './data/')
    parser.add_argument('--logpath', default='./logs/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--date',default='0321')
    parser.add_argument('--plot',default=True)
    args = parser.parse_args()
    main(args)
