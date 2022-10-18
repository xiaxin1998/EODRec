import os
import time
import torch
import argparse
import numpy as np
from student_pretrain import *
from utils import *
import pickle
from sas import SAS
from module import *

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='retailrocket')
parser.add_argument('--act', default='gelu')
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--num_layer', default=1, type=int)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=256, type=int)
parser.add_argument('--inner_units', default=256, type=int)
parser.add_argument('--num_blocks', default=1, type=int)
parser.add_argument('--num_heads', default=2, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--beta', type=float, default=0.0, help='the scale of cl')
parser.add_argument('--gamma', type=float, default=0.0, help='the scale of kd')
parser.add_argument('--eta', type=float, default=0.3)
parser.add_argument('--hidden_dim',      type=int,   default=256,      help='number of dimensions of input size')
parser.add_argument('--code_book_len',   type=int,   default=4,       help='number of codebooks')
parser.add_argument('--cluster_num',     type=int,   default=32,       help='length of a codebook')
parser.add_argument('--coe', type=float, default=0., help='the scale of mse')
parser.add_argument('--seed', type=int, default=55)
opt = parser.parse_args()
print(opt)


def main():
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    if opt.dataset == 'Tmall':
        n_node = 40727+2
    elif opt.dataset == 'retailrocket':
        n_node = 36972
    else:
        n_node = 309 + 1
    train_data = Data(train_data, train_data, shuffle=True, n_node=n_node, train=1)
    test_data = Data(test_data, train_data, shuffle=False, n_node=n_node, train=0)
    model = SASRec(n_node, opt)
    model = trans_to_cuda(model)
    path_state_dict = "../sas_teacher_rr_nn.pkl"
    state = torch.load(path_state_dict)
    teacher = SAS(n_node, opt)
    teacher.load_state_dict(state)

    emb_model = CODE_AE(opt)

    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0]
        best_results['nDCG%d' % K] = [0, 0, 0]
    for epoch in range(200):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data, epoch, opt, trans_to_cuda(teacher), trans_to_cuda(emb_model))
        # torch.save(model.one_hot.cpu(), '../onehot_tmall.pkl')
        if epoch == 0:
            torch.save(trans_to_cpu(model.one_hot).detach().numpy(), '../sas_stu_rr_onehot_4m_256k_test03.pkl')
            torch.save(emb_model.state_dict(), '../code_model_rr_4m_256k_test03.pkl')
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            metrics['nDCG%d' % K] = np.mean(metrics['nDCG%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
            if best_results['metric%d' % K][2] < metrics['nDCG%d' % K]:
                best_results['metric%d' % K][2] = metrics['nDCG%d' % K]
                best_results['epoch%d' % K][2] = epoch
        print(metrics)
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tnDCG%d: %.4f\tEpoch: %d,  %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1], K,
                   best_results['metric%d' % K][2],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1], best_results['epoch%d' % K][2]))


if __name__ == '__main__':
    main()
