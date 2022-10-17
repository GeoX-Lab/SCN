import torch.nn as nn
import torch


class integration_network(nn.Module):

    def __init__(self, feature_extractor, task_num):
        super(integration_network, self).__init__()
        self.feature = feature_extractor
        self.fc = nn.Linear(feature_extractor.fc.in_features, task_num, bias=True)
        self.fc_list = []
        self.task_num = task_num

    def forward(self, input):
        x, feature_list = self.feature(input)
        x1 = self.fc(x)
        out_list = []
        for i in range(len(self.fc_list)):
            # out_list += self.fc_list[i](x)
            if len(out_list) is 0:
                out_list = self.fc_list[i](x)
            else:
                out_list = torch.cat([out_list, self.fc_list[i](x)], dim=1)
        return x1, feature_list, out_list

    def Incremental_learning(self):
        self.fc = nn.Linear(self.feature.fc.in_features, self.task_num, bias=True)

    def fc_append(self):
        self.fc_list.append(self.fc)


class test_network(nn.Module):

    def __init__(self, feature_extractor, fc):
        super(test_network, self).__init__()
        self.feature = feature_extractor
        self.fc = fc

    def forward(self, input):
        x, feature_list = self.feature(input)
        x = self.fc(x)
        return x, feature_list
