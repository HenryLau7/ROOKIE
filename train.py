import numpy as np
from utils import cross_entropy_loss
import pickle

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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
        
    
def batch_norm(x, epsilon=1e-5):
    # 计算每个特征的均值和方差
    mean = np.mean(x, axis=0)
    variance = np.var(x, axis=0)
    
    # 归一化
    X_normalized = (x - mean) / np.sqrt(variance + epsilon)

    return X_normalized

class trainer:
    def __init__(self, model, optimizer, epoch, minibatch_train, minibatch_val, save_dir):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.epochs_per_vali = 2
        self.minibatch_train = minibatch_train
        self.minibatch_val = minibatch_val
        self.criterion = cross_entropy_loss
        
        self.epochs_per_vali = 5
        self.epochs_decay = 200
        
        self.best_acc = 0
        self.save_dir = save_dir
        
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train(self):
        
        for epoch in range(self.epoch):
            total = 0
            correct = 0
            loss_meter = AverageMeter()
            
            for minibatch_x, minibatch_y in self.minibatch_train:
                total += minibatch_x.shape[0]
                minibatch_x = batch_norm(minibatch_x)
                output, cache = self.model(minibatch_x)
                loss = self.criterion(minibatch_y,output) / minibatch_x.shape[0]
                grads = self.model.backward(minibatch_x,minibatch_y,cache)
                self.optimizer.step(grads)
                prediction = np.argmax(output,axis=1)
                gt = np.argmax(minibatch_y, axis=1)
                correct += np.sum(prediction == gt)
                # print(correct)
                loss_meter.update(loss.item())
            
            acc = 100.*correct/total
            self.train_losses.append(loss_meter.avg)
            self.train_accuracies.append(acc)
            print("Epoch:{}, Loss:{}, Acc:{}".format(epoch, loss_meter.avg, acc))
            
            if (epoch+1) % self.epochs_per_vali == 0:
                val_loss, val_acc = self.validation(epoch)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
            if (epoch+1) % self.epochs_decay == 0:
                self.optimizer.adjust_learning_rate()
                print('learning rate decay:{}'.format(self.optimizer.learning_rate))
                
    def validation(self, epoch):
        total = 0
        correct = 0
        loss_meter = AverageMeter()
        for minibatch_x, minibatch_y in self.minibatch_val:
            total += minibatch_x.shape[0]
            minibatch_x = batch_norm(minibatch_x)
            output, _ = self.model(minibatch_x)
            loss = self.criterion(minibatch_y,output) / minibatch_x.shape[0]
            
            prediction = np.argmax(output,axis=1)
            gt = np.argmax(minibatch_y, axis=1)
            correct += np.sum(prediction == gt)
            loss_meter.update(loss.item())
            
        acc = 100.*correct/total
        print("Validation: Loss:{}, Acc:{}".format(loss_meter.avg, acc))
        
        if acc > self.best_acc:
            self.best_acc = acc
            print('Best model updated!')
            checkpoints = {'parameters':self.model.parameters,
                           'acc':acc,
                           'epoch':epoch
            }
            with open(self.save_dir + 'best_model.pkl', 'wb') as f:
                pickle.dump(checkpoints, f)
        
        return loss_meter.avg, acc
                
def test(model, minibatch_test):
    total = 0
    correct = 0
    for minibatch_x, minibatch_y in minibatch_test:
        total += minibatch_x.shape[0]
        minibatch_x = batch_norm(minibatch_x)
        output, _ = model(minibatch_x)
        
        prediction = np.argmax(output,axis=1)
        gt = np.argmax(minibatch_y, axis=1)
        correct += np.sum(prediction == gt)
        
    acc = 100.*correct/total
    print("Test: Acc:{}".format(acc))
    