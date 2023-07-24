from argparse import ArgumentParser

from data_loader import load_data

import torch
import torch.nn as nn

# from model import GCN
from aug import aug
from model import Grace
from eval import label_classification
# from model import YourGNNModel # Build your model in model.py
    
import os
import warnings
warnings.filterwarnings("ignore")

def evaluate(g, features, labels, mask, model):
    """Evaluate model accuracy"""
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def train(g, features, train_labels, val_labels, train_mask, val_mask, model, epochs,  device, es_iters=None):
    
    # define train/val samples, loss function and optimizer
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    
    drop_edge_rate_1 = 0.4
    drop_edge_rate_2 = 0.1
    drop_feature_rate_1 = 0.1
    drop_feature_rate_2 = 0.2

#     drop_edge_rate_1 = 0.3
#     drop_edge_rate_2 = 0.3
#     drop_feature_rate_1 = 0.3
#     drop_feature_rate_2 = 0.3


    # If early stopping criteria, initialize relevant parameters
    if es_iters:
        print("Early stopping monitoring on")
        loss_min = 1e8
        es_i = 0

    # training loop
    for epoch in range(epochs):
        model.train()
        
        graph1, feat1 = aug(g, features, drop_feature_rate_1, drop_edge_rate_1)
        graph2, feat2 = aug(g, features, drop_feature_rate_2, drop_edge_rate_2)

        graph1 = graph1.to(device)
        graph2 = graph2.to(device)

        feat1 = feat1.to(device)
        feat2 = feat2.to(device)
        
        
        
        
#         logits = model(g, features)
        loss = model(graph1, graph2, feat1, feat2)
#         loss = loss_fcn(logits[train_mask], train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#         acc = evaluate(g, features, val_labels, val_mask, model)
        
        print(f"Epoch={epoch:03d}, loss={loss.item():.4f}")
#         print(
#             "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
#                 epoch, loss.item(), acc
#             )
#         )
        
#         val_loss = loss_fcn(logits[val_mask], val_labels).item()
#         if es_iters:
#             if val_loss < loss_min:
#                 loss_min = val_loss
#                 es_i = 0
#             else:
#                 es_i += 1

#             if es_i >= es_iters:
#                 print(f"Early stopping at epoch={epoch+1}")
#                 break


if __name__ == '__main__':

    parser = ArgumentParser()
    # you can add your arguments if needed
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--es_iters', type=int, help='num of iters to trigger early stopping', default=300)
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    if args.use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print("device: "+str(device))

    # Load data
#     features, graph, num_classes, \
#     train_labels, val_labels, test_labels, \
#     train_mask, val_mask, test_mask = load_data()
    
    features, graph, num_classes, \
    val_labels, train_labels, test_labels, \
    val_mask, train_mask, test_mask = load_data()
    
    print(len(train_labels))
    print(len(val_labels))
    print(len(test_labels))
    
    # Initialize the model (Baseline Model: GCN)
    """TODO: build your own model in model.py and replace GCN() with your model"""
    in_size = features.shape[1]
    out_size = num_classes
#     model = GCN(in_size, 16, out_size).to(device)
    hid_dim = 256
    out_dim = 256
    num_layers = 2
    act_fn = nn.ReLU()
#     act_fn = nn.PReLU()
    temp = 0.7
    model = Grace(in_size, hid_dim, out_dim, num_layers, act_fn, temp)
    model = model.to(device)
    path_name='model.pth'
    
    # model training
    print("Training...")
    train(graph, features, train_labels, val_labels, train_mask, val_mask, model, args.epochs, device, args.es_iters)
    torch.save(model, path_name)
    print("=========save model=========")

#     model=torch.load(path_name)
#     model=model.to(device)

    print("Testing...")
    model.eval()
#     with torch.no_grad():
#         logits = model(graph, features)
#         logits = logits[test_mask]
#         _, indices = torch.max(logits, dim=1)


    graph = graph.add_self_loop()
    graph = graph.to(device)
    features = features.to(device)
    embeds = model.get_embedding(graph, features)

    """Evaluation Embeddings  """
    indices = label_classification(
        embeds, train_labels, val_labels, train_mask, val_mask, test_mask
    )

#     torch.cat((train_labels,val_labels), 0)
#     print("Export predictions as csv file.")
#     with open('output.csv', 'w') as f:
#         f.write('Id,Predict\n')
#         for idx, pred in enumerate(indices):
#             f.write(f'{idx},{int(pred)}\n')
    # Please remember to upload your output.csv file to Kaggle for scoring