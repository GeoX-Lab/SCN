import utils
from torch.nn import functional as F
import torch.optim as optim

from datasets import iUCMerced, iAID
from datasets.iNWPU import iNWPU
from datasets.iRSICB256 import iRSICB256
from myNetwork import integration_network, test_network
from Long_ShortNet import long_short_network

from torch.utils.data import DataLoader

from shortnets import pyconvresnet18, resnet18_cbam, resnet50_cbam
from shortnets.Res2Net import res2net50
from shortnets.ResNeSt import resnest50
from shortnets.dcd import resnet18_dcd
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_one_hot(target, num_class):
    one_hot = torch.zeros(target.shape[0], num_class).to(device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot


class TMN_M_H_model:

    def __init__(self, method, parse, integration_net, batch_size, epochs, integration_lr, short_lr,
                 temp, dataset='UCMerced', img_size=32, short_net='resnet18'):
        self.distill_method = 'cross_similar'
        self.gama = 30
        self.beta_distill = 0.08

        self.method = method
        self.epochs = epochs
        self.integration_lr = integration_lr
        self.short_lr = short_lr
        self.temp = temp

        self.exemplar_set = []

        self.long_model = None
        self.l_short_model = None

        # determine which kinds of net do short need
        self.short_net = short_net

        # set datasets
        if dataset == 'RSICB256':
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # todo :not clc yet
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            self.test_transform = transforms.Compose([
                transforms.RandomCrop(img_size, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            self.train_dataset = iRSICB256('../../dataset/RSI-CB256', train=True,
                                           train_transform=self.train_transform)
            self.test_dataset = iRSICB256('../../dataset/RSI-CB256', train=False, test_transform=self.test_transform)

        if dataset == 'UCMerced':
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            self.test_transform = transforms.Compose([
                transforms.RandomCrop(img_size, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            self.train_dataset = iUCMerced('../../dataset/UCMerced_LandUse', train=True,
                                           train_transform=self.train_transform)
            self.test_dataset = iUCMerced('../../dataset/UCMerced_LandUse', train=False,
                                          test_transform=self.test_transform)
        if dataset == 'AID':
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # TODO: not clc yet
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            self.test_transform = transforms.Compose([
                transforms.RandomCrop(img_size, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            self.train_dataset = iAID('../../dataset/AID', train=True,
                                      train_transform=self.train_transform)
            self.test_dataset = iAID('../../dataset/AID', train=False, test_transform=self.test_transform)
        if dataset == 'NWPU':
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # TODO: not clc yet
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            self.test_transform = transforms.Compose([
                transforms.RandomCrop(img_size, padding=4),
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            self.train_dataset = iNWPU('../../dataset/NWPU-RESISC45', train=True,
                                      train_transform=self.train_transform)
            self.test_dataset = iNWPU('../../dataset/NWPU-RESISC45', train=False, test_transform=self.test_transform)

        self.numclass = int(self.train_dataset.class_num / parse)
        self.task_size = int(self.train_dataset.class_num / parse)
        self.integration_model = integration_network(integration_net, self.task_size)
        self.integration_model.fc_list = []
        self.batchsize = batch_size

        self.train_loader = None
        self.test_loader = None
        self.current_task_num = 0

    # get incremental train data
    # incremental
    def beforeTrain(self):
        self.current_task_num += 1
        classes = [self.numclass - self.task_size, self.numclass]
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        if self.numclass > self.task_size:
            self.l_short_model = None
            if self.short_net == 'pyconv':
                self.l_short_model = long_short_network(self.long_model, self.method, self.task_size,
                                                        feature_extractor=pyconvresnet18())
            elif self.short_net == 'FPN':
                raise NotImplementedError("Not available the pretrained model yet!")
                # self.l_short_model = long_short_network(self.long_model, self.method,
                #                                         feature_extractor=fpn_resnet18_cbam())
            elif self.short_net == 'dcd':
                self.l_short_model = long_short_network(self.long_model, self.method, self.task_size,
                                                        feature_extractor=resnet18_dcd())
            elif self.short_net == 'resnest':
                self.l_short_model = long_short_network(self.long_model, self.method, self.task_size,
                                                        feature_extractor=resnest50())
            elif self.short_net == 'res2net':
                self.l_short_model = long_short_network(self.long_model, self.method, self.task_size,
                                                        feature_extractor=res2net50())
            else:
                print('default short-net: RESNET50 (TMN-v1)')
                self.l_short_model = long_short_network(self.long_model, self.method, self.task_size,
                                                        feature_extractor=resnet50_cbam())

            self.integration_model.Incremental_learning()

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes, self.exemplar_set)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.batchsize)

        return train_loader, test_loader

    # train model;  compute loss;   evaluate model
    def train_integration(self):
        if self.l_short_model is not None:
            self.l_short_model.eval()
        self.integration_model.eval()
        self.integration_model.train()
        self.integration_model.to(device)

        inte_acc_list = []
        opt = optim.SGD(self.integration_model.parameters(), lr=self.integration_lr, momentum=0.9, nesterov=True,
                        weight_decay=0.00001)
        for epoch in range(self.epochs):
            if epoch == 40:
                opt = optim.SGD(self.integration_model.parameters(), lr=self.integration_lr / 5, momentum=0.9,
                                nesterov=True, weight_decay=0.00001)
                print("change learning rate%.3f" % (self.integration_lr / 5))
            elif epoch == 60:
                opt = optim.SGD(self.integration_model.parameters(), lr=self.integration_lr / 25, momentum=0.9,
                                nesterov=True, weight_decay=0.00001)
                print("change learning rate%.5f" % (self.integration_lr / 25))
            elif epoch == 70:
                opt = optim.SGD(self.integration_model.parameters(), lr=self.integration_lr / 125, momentum=0.9,
                                nesterov=True, weight_decay=0.00001)
                print("change learning rate%.5f" % (self.integration_lr / 50))
            accuracy = self._test(self.test_loader, test_phase='integration')
            inte_acc_list.append(accuracy)
            print('integration_epoch:%d,integration_accuracy:%.5f' % (epoch, accuracy))
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(device), target.to(device)
                opt.zero_grad()
                loss = self._compute_loss(images, target, training_phase='integration')
                loss.backward()
                opt.step()
                print('integration_epoch:%d,step:%d,loss:%.3f' % (epoch, step, loss.item()))
        # save acc of integration
        file_path = 'result/acc_intenet%s_%d.txt' % (self.method, self.numclass)
        filename = open(file_path, 'w')
        for value in inte_acc_list:
            filename.write(str(value))
        filename.close()

        # plot acc
        fig_path = 'result/acc_intenet%s_%d.png' % (self.method, self.numclass)
        plot_acc_one_task(inte_acc_list, fig_path)

    def train_short(self):
        self.integration_model.eval()

        self.l_short_model.eval()
        self.l_short_model.train()
        self.l_short_model.to(device)

        short_loss_list = []
        short_acc_list = []
        opt = optim.SGD(self.l_short_model.parameters(), lr=self.short_lr, momentum=0.9, nesterov=True,
                        weight_decay=0.00001)
        for epoch in range(self.epochs):
            if epoch == 48:
                opt = optim.SGD(self.l_short_model.parameters(), lr=self.short_lr / 5, momentum=0.9, nesterov=True,
                                weight_decay=0.00001)
                print("change learning rate%.3f" % (self.short_lr / 5))
            elif epoch == 88:
                opt = optim.SGD(self.l_short_model.parameters(), lr=self.short_lr / 25, momentum=0.9, nesterov=True,
                                weight_decay=0.00001)
                print("change learning rate%.5f" % (self.short_lr / 25))
            elif epoch == 125:
                opt = optim.SGD(self.l_short_model.parameters(), lr=self.short_lr / 125, momentum=0.9, nesterov=True,
                                weight_decay=0.00001)
                print("change learning rate%.5f" % (self.short_lr / 125))

            accuracy = self._test(self.test_loader, test_phase='short')
            short_acc_list.append(accuracy)
            print('short_epoch:%d,short_accuracy:%.5f' % (epoch, accuracy))
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(device), target.to(device)
                opt.zero_grad()
                loss = self._compute_loss(images, target, training_phase='short')
                opt.zero_grad()
                loss.backward()
                opt.step()
                print('short_epoch:%d,step:%d,loss:%.3f' % (epoch, step, loss.item()))
                short_loss_list.append(loss.item())
        # save acc&loss
        file_path = 'result/loss_shortnet%s_%d.txt' % (self.method, self.numclass)
        filename = open(file_path, 'w')
        for value in short_loss_list:
            filename.write(str(value))
        filename.close()

        file_path = 'result/acc_shortnet%s_%d.txt' % (self.method, self.numclass)
        filename = open(file_path, 'w')
        for value in short_acc_list:
            filename.write(str(value))
        filename.close()
        # plot acc&loss
        fig_path = 'result/acc_shortnet%s_%d.png' % (self.method, self.numclass)
        plot_acc_one_task(short_acc_list, fig_path)
        fig_path = 'result/loss_shortnet%s_%d.png' % (self.method, self.numclass)
        plot_loss_one_task(short_loss_list, fig_path)

    def _test(self, testloader, test_phase, fc=None, i=None):
        if test_phase == 'integration':
            self.integration_model.eval()
            if fc is None:
                test_model = self.integration_model
            else:
                test_model = test_network(self.integration_model.feature, fc)
            test_model.eval()
            correct, total = 0.0, 0.0
            for setp, (indexs, imgs, labels) in enumerate(testloader):
                if i == None:
                    labels = labels - self.numclass + self.task_size
                else:
                    labels = labels - self.task_size * i
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.no_grad():
                    if fc is None:
                        outputs, _, _ = test_model(imgs)
                    else:
                        outputs, _ = test_model(imgs)
                predicts = torch.max(outputs, dim=1)[1]
                correct += (predicts.cpu() == labels.cpu()).sum()
                total += len(labels)

            accuracy = correct.item() / total
            self.integration_model.train()
            return accuracy
        elif test_phase == 'short':
            self.l_short_model.eval()
            correct, total = 0.0, 0.0
            for setp, (indexs, imgs, labels) in enumerate(testloader):
                labels = labels - self.numclass + self.task_size
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.no_grad():
                    outputs, _ = self.l_short_model(imgs)
                predicts = torch.max(outputs, dim=1)[1]
                correct += (predicts.cpu() == labels.cpu()).sum()
                total += len(labels)

            accuracy = correct.item() / total
            self.l_short_model.train()

            return accuracy

    def confusion_matrix(self, preds, lables, conf_matrix):
        for p, t in zip(preds, lables):
            conf_matrix[p, t] += 1
        return conf_matrix

    def _clc_confusion_matrix(self):
        conf_matrix = torch.zeros(self.numclass, self.numclass)
        classes = [0, self.current_task_num * self.task_size]
        _, test_loader = self._get_train_and_test_dataloader(classes)

        for setp, (indexs, imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                if self.current_task_num is 1:
                    output, _, _ = self.integration_model(imgs)
                else:
                    _, _, output = self.integration_model(imgs)
            predicts = torch.max(output, dim=1)[1]
            conf_matrix = self.confusion_matrix(predicts, labels, conf_matrix)

        utils.plot_confusion_matrix(conf_matrix, self.current_task_num, self.train_dataset.class_num)

        return

    def _test_all_tasks(self):
        acc_all_tasks = []
        tasks_num = int(self.numclass / self.task_size)
        for i in range(tasks_num):
            training_phase = 'integration'
            classes = [self.task_size * i, self.task_size * i + self.task_size]
            _, test_loader = self._get_train_and_test_dataloader(classes)
            if i == (tasks_num - 1):
                accuracy = self._test(test_loader, training_phase, self.integration_model.fc)
            else:
                accuracy = self._test(test_loader, training_phase, self.integration_model.fc_list[i], i)
            print('test accuracy of task_%d = %.5f' % ((i + 1), accuracy))
            acc_all_tasks.append(accuracy)
        avg_acc = sum(acc_all_tasks) / tasks_num
        acc_all_tasks.append(avg_acc)
        # save acc of integration
        file_path = 'result/acc_all_intenet%s_%d.txt' % (self.method, self.numclass)
        filename = open(file_path, 'w')
        for value in acc_all_tasks:
            filename.write(str(value))
        filename.close()

        # plot conf_matrix

        return accuracy

    def _compute_loss(self, imgs, target, training_phase):
        if training_phase == 'short':
            output, _ = self.l_short_model(imgs)
            target = target - self.numclass + self.task_size
            target = get_one_hot(target, self.task_size)
            output, target = output.to(device), target.to(device)
            return F.binary_cross_entropy_with_logits(output, target)
        elif training_phase == 'integration':
            output, features_inte, out_list = self.integration_model(imgs)
            target = target - self.numclass + self.task_size
            target = get_one_hot(target, self.task_size)
            if self.long_model == None:
                output, target = output.to(device), target.to(device)
                return F.binary_cross_entropy_with_logits(output, target)
            else:
                return self.mixed_distill_loss(output, out_list, target, imgs, features_inte)

    # mixed distillation loss of S-net to I-net
    def mixed_distill_loss(self, output, out_list, target, imgs, features_inte):
        if self.distill_method == 'cross':
            with torch.no_grad():
                _, _, out_list_long = self.long_model(imgs)
            distill_long_loss = self.cal_distill_long(out_list, out_list_long)
            new_task_loss = F.binary_cross_entropy_with_logits(output, target)
            return new_task_loss + distill_long_loss
        elif self.distill_method == 'sigmoid':
            with torch.no_grad():
                _, _, out_list_long = self.long_model(imgs)
                target_short, _ = self.l_short_model(imgs)
            distill_long_loss = self.cal_distill_long(out_list, out_list_long)
            new_task_loss = (1 - self.beta_distill) * F.binary_cross_entropy_with_logits(output, target) + \
                            self.beta_distill * self.temp ** 2 * F.binary_cross_entropy(tempsigmoid(output, self.temp),
                                                                                        tempsigmoid(target_short,
                                                                                                    self.temp))
            # sigmoid + cross
            return new_task_loss + distill_long_loss
        elif self.distill_method == 'cross_similar':
            with torch.no_grad():
                _, _, out_list_long = self.long_model(imgs)
                _, features_short = self.l_short_model(imgs)
            distill_long_loss = self.cal_distill_long(out_list, out_list_long)
            new_task_loss = F.binary_cross_entropy_with_logits(output, target)
            # similar + cross
            return self.gama * self.similarity_loss(features_short, features_inte) + new_task_loss + distill_long_loss

    def cal_distill_long(self, out_ls, out_ls_long):
        distill_long = 0
        for i in range(len(out_ls_long)):
            distill_long += F.binary_cross_entropy(tempsigmoid(out_ls[i], self.temp),
                                                   tempsigmoid(out_ls_long[i], self.temp))
        return distill_long

    # cal similarity loss between teacher-net and student-net
    def similarity_loss(self, features_short, features_inte):
        losses = []
        for k in range(len(features_inte)):
            f_s = features_inte[k]
            f_t = features_short[k]
            bsz = f_s.shape[0]
            f_s = f_s.view(bsz, -1)
            f_t = f_t.view(bsz, -1)

            G_s = torch.mm(f_s, torch.t(f_s))
            # G_s = G_s / G_s.norm(2)
            G_s = torch.nn.functional.normalize(G_s)
            G_t = torch.mm(f_t, torch.t(f_t))
            # G_t = G_t / G_t.norm(2)
            G_t = torch.nn.functional.normalize(G_t)

            G_diff = G_t - G_s
            loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
            losses.append(loss)
        return sum(losses)

    def afterTrain(self, paint_cf=False):
        self.integration_model.fc_append()
        self._test_all_tasks()  # test all tasks
        if paint_cf:
            self._clc_confusion_matrix()
        self.numclass += self.task_size
        filename = 'model/integration_increment:%d_net.pkl' % (self.numclass - self.task_size)
        torch.save(self.integration_model, filename)

        self.long_model = torch.load(filename)

        self.long_model.to(device)
        self.long_model.eval()
