#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2024/4/17 20:57
"""
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet

def load_ESOL(args):
    # 每个样本：Data(x=[32, 9], edge_index=[2, 68], edge_attr=[68, 3], smiles='OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O ', y=[1, 1])
    dataset = MoleculeNet(root='Data/MoleculeNet', name='ESOL')

    # 1128个样本用于graph-level prediction 训练：902；测试：226
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=6)

    return train_loader, test_loader, dataset.num_node_features, dataset.num_edge_features
