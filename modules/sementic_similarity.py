import torch
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
# import vit_pytorch.vit_with_patch_merger as vit
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CooccurrenceMatrix(nn.Module):
    def __init__(self):
        super(CooccurrenceMatrix, self).__init__()
        # 进行相似性矩阵的重新加权，这里假设λ = 0.5
        self.c = nn.Parameter(torch.tensor(1.0), requires_grad=True)  # concepts的数量
        self.lambda_value = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 可训练的 lambda_value 参数

    def forward(self, labels):

        batch_size, num_labels = labels.size()
        torch.set_printoptions(profile="full")

        # 计算标签共现矩阵 Ni,j，使用内积计算
        cooccurrence_matrix = torch.matmul(labels.float(), labels.t().float())

        smoothing_term = 1e-6  # 可以根据实际情况调整平滑项的大小
        #labels = labels.cpu().numpy()
        sum_labels = torch.sum(labels, dim=1, keepdim=True)
        # 计算每个 Ni,j 对应的初始相似性矩阵 Mi,j'
        cooccurrence_matrix = cooccurrence_matrix / (sum_labels.float() + smoothing_term )
        return cooccurrence_matrix

    # def visualize(self, cooccurrence_matrix):
    #     plt.figure(figsize=(10, 8))
    #     plt.imshow(cooccurrence_matrix.cpu(), cmap='hot', interpolation='nearest')
    #     plt.title('Cooccurrence Matrix Heatmap')
    #     plt.colorbar()
    #     plt.show()



class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)  # (batch_size, out_features)
        output = torch.mm(adj, support)        # (batch_size, out_features)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, in_features, nhid, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(in_features, nhid)
        self.dropout = dropout

    def forward(self, image_features, adj_matrix):
        degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))
        degree_matrix = degree_matrix + 1e-5 * torch.eye(degree_matrix.size(0), device=degree_matrix.device)
        normalized_adjacency_matrix = torch.mm(torch.mm(torch.inverse(torch.sqrt(degree_matrix)), adj_matrix),
                                               torch.inverse(torch.sqrt(degree_matrix)))
        x = torch.nn.functional.relu(self.gc1(image_features, normalized_adjacency_matrix))
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        return x

class SemanticSimilarityLearning(nn.Module):
    def __init__(self, input_dim,  num_gcn_layers):
        super(SemanticSimilarityLearning, self).__init__()
        self.num_gcn_layers = num_gcn_layers
        self.co_occurrence_matrix = CooccurrenceMatrix()
        self.gcn_layers = nn.ModuleList([GCN(in_features=input_dim, nhid=16,dropout=0.2) for _ in range(num_gcn_layers)]).to(device)


    def forward(self, feats, labels):
        h = feats

        co_occurrence_matrix = self.co_occurrence_matrix(labels)#[bs,bs]
        for gcn_layer in self.gcn_layers:
            h = gcn_layer(h, co_occurrence_matrix)
            h = F.leaky_relu(h)

        return h

