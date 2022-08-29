import argparse
import json
import logging
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.loader import NeighborLoader

# from torch_geometric.nn import RGCNConv
from tqdm import tqdm

from models import RGINConv as RGCNConv

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# 参数
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="data/icdm.pt")
parser.add_argument("--testset", type=str, default="data/icdm2022_session2.pt")
parser.add_argument("--labeled-class", type=str, default="item")
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    help="Mini-batch size. If -1, use full graph training.",
)
parser.add_argument(
    "--fanout", type=int, default=150, help="Fan-out of neighbor sampling."
)
parser.add_argument(
    "--n-layers", type=int, default=3, help="number of propagation rounds"
)
parser.add_argument("--h-dim", type=int, default=256, help="number of hidden units")
parser.add_argument("--in-dim", type=int, default=256, help="number of hidden units")
parser.add_argument(
    "--n-bases",
    type=int,
    default=8,
    help="number of filter weight matrices, default: -1 [use all]",
)
parser.add_argument("--validation", type=bool, default=True)
parser.add_argument("--early_stopping", type=int, default=6)
parser.add_argument("--n-epoch", type=int, default=100)
parser.add_argument(
    "--test-file", type=str, default="data/icdm2022_session2_test_ids.txt"
)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--comment", type=str, default="")
parser.add_argument("--device_id", type=str, default="0")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hgraph = torch.load(args.dataset)

labeled_class = args.labeled_class

train_idx = hgraph[labeled_class].pop("train_idx")
if args.validation:
    val_idx = hgraph[labeled_class].pop("val_idx")
# Mini-Batch

train_loader = NeighborLoader(
    hgraph,
    input_nodes=(labeled_class, train_idx),
    num_neighbors=[args.fanout] * args.n_layers,
    shuffle=True,
    batch_size=args.batch_size,
)

if args.validation:
    val_loader = NeighborLoader(
        hgraph,
        input_nodes=(labeled_class, val_idx),
        num_neighbors=[args.fanout] * args.n_layers,
        shuffle=False,
        batch_size=args.batch_size,
    )

# session2 test dataset
testgraph = torch.load(args.testset)        #testset
test_id = [int(x) for x in open(args.test_file).readlines()]
converted_test_id = []
for i in test_id:
    converted_test_id.append(testgraph["item"].maps[i])
test_idx = torch.LongTensor(converted_test_id)

test_loader = NeighborLoader(
    testgraph,
    input_nodes=(labeled_class, test_idx),
    num_neighbors=[args.fanout] * args.n_layers,
    shuffle=False,
    batch_size=args.batch_size,
)

# # No need to maintain these features during evaluation:
# # Add global node index information.
# test_loader.data.num_nodes = data.num_nodes
# test_loader.data.n_id = torch.arange(data.num_nodes)


num_relations = len(hgraph.edge_types)


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = F.relu
        self.convs.append(
            RGCNConv(
                in_channels,
                hidden_channels,
                num_relations,
                num_blocks=args.n_bases,
                aggr="max",
            )
        )
        for i in range(n_layers - 2):
            self.convs.append(
                RGCNConv(
                    hidden_channels,
                    hidden_channels,
                    num_relations,
                    num_blocks=args.n_bases,
                    aggr="max",
                )
            )
        self.convs.append(
            RGCNConv(
                hidden_channels,
                out_channels,
                num_relations,
                num_bases=args.n_bases,
                aggr="max",
            )
        )

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.4, training=self.training)
        return x


model = RGCN(
    in_channels=args.in_dim,
    hidden_channels=args.h_dim,
    out_channels=2,
    n_layers=args.n_layers,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
criterion = nn.CrossEntropyLoss()
criterion.cuda()
# criterion = nn.CrossEntropyLoss()


def train(epoch):
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)), ascii=True)
    pbar.set_description(f"Epoch {epoch:02d}")

    total_loss = total_correct = total_examples = 0
    y_pred = []
    y_true = []
    for batch in train_loader:
        optimizer.zero_grad()
        batch_size = batch[labeled_class].batch_size
        y = batch[labeled_class].y[:batch_size].to(device)

        # 找到应该输出的index起始值,因为item节点没有放置在最前面，导致to_homogeneous后source node并不会排在最前面
        start = 0
        for ntype in batch.node_types:
            if ntype == labeled_class:
                break
            start += batch[ntype].num_nodes

        batch = batch.to_homogeneous()

        y_hat = model(
            batch.x.to(device), batch.edge_index.to(device), batch.edge_type.to(device)
        )[start : start + batch_size]
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
        y_true.append(y.cpu())
        total_loss += float(loss) * batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch_size
        pbar.update(batch_size)
        torch.cuda.empty_cache()
    pbar.close()
    ap_score = average_precision_score(
        torch.hstack(y_true).numpy(), torch.hstack(y_pred).numpy()
    )

    return total_loss / total_examples, total_correct / total_examples, ap_score


@torch.no_grad()
def val():
    model.eval()
    pbar = tqdm(total=int(len(val_loader.dataset)), ascii=True)
    pbar.set_description(f"Epoch {epoch:02d}")
    total_loss = total_correct = total_examples = 0
    y_pred = []
    y_true = []
    for batch in val_loader:
        batch_size = batch[labeled_class].batch_size
        y = batch[labeled_class].y[:batch_size].to(device)
        # 找到应该输出的index起始值,因为item节点没有放置在最前面，导致to_homogeneous后source node并不会排在最前面，而是排列在给类别最前面
        start = 0
        for ntype in batch.node_types:
            if ntype == labeled_class:
                break
            start += batch[ntype].num_nodes

        batch = batch.to_homogeneous()

        y_hat = model(
            batch.x.to(device), batch.edge_index.to(device), batch.edge_type.to(device)
        )[start : start + batch_size]
        loss = F.cross_entropy(y_hat, y)
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
        y_true.append(y.cpu())
        total_loss += float(loss) * batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch_size
        pbar.update(batch_size)
        torch.cuda.empty_cache()
    pbar.close()
    ap_score = average_precision_score(
        torch.hstack(y_true).numpy(), torch.hstack(y_pred).numpy()
    )

    return total_loss / total_examples, total_correct / total_examples, ap_score


@torch.no_grad()
def test():
    model.eval()
    pbar = tqdm(total=int(len(test_loader.dataset)), ascii=True)
    pbar.set_description(f"Generate Final Result:")
    y_pred = []
    for batch in test_loader:
        batch_size = batch[labeled_class].batch_size
        # 找到应该输出的index,因为item节点没有放置在最前面，导致to_homogeneous后source node并不会排在最前面，而是排列在给类别最前面
        start = 0
        for ntype in batch.node_types:
            if ntype == labeled_class:
                break
            start += batch[ntype].num_nodes

        # 转换为同质图
        batch = batch.to_homogeneous()
        y_hat = model(
            batch.x.to(device), batch.edge_index.to(device), batch.edge_type.to(device)
        )[start : start + batch_size]
        pbar.update(batch_size)
        y_pred.append(F.softmax(y_hat, dim=1)[:, 1].detach().cpu())
    pbar.close()

    return torch.hstack(y_pred)


base_dir = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
resultpath = "results/" + base_dir
file_name = os.path.join(resultpath, "results.txt")
model_path = os.path.join(resultpath, "best_mode.pt")

if not os.path.exists(resultpath):
    os.makedirs(resultpath)
# set_logger(logger, file_name)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s: - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

fh = logging.FileHandler(file_name)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

logger.info("Start train and use device {}".format(device))
logger.info(args)

val_ap_list = []
best_loss = 1e10
end = 0
best_epoch, best_ap = 0, 0
count = 0
for epoch in range(1, args.n_epoch + 1):
    train_loss, train_acc, train_ap = train(epoch)
    logger.info(
        f"Train: Epoch {epoch:02d}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AP_Score: {train_ap:.4f}"
    )

    val_loss, val_acc, val_ap = val()
    logger.info(
        f"Val: Epoch: {epoch:02d}, Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AP_Score: {val_ap:.4f}"
    )
    if val_ap > best_ap:
        best_loss, count, best_epoch, best_ap = val_loss, 0, epoch, val_ap
        torch.save(model, model_path)
        logger.info("save model in epoch {}".format(epoch))
    else:
        count += 1

    if count > args.early_stopping:
        logger.info(
            "Early Stopping best epoch is {} best loss is {} best ap is {}".format(
                best_epoch, best_loss, best_ap
            )
        )
        break

    val_ap_list.append(float(val_ap))
    ave_val_ap = np.average(val_ap_list)
    end = epoch

logger.info("Start infer...")
model = torch.load(model_path)
y_pred = test()
with open(
    resultpath + "/pyg_pred_{:.5f}_{:.5f}.json".format(best_loss, best_ap), "w+"
) as f:
    for i in range(len(test_id)):
        y_dict = {}
        y_dict["item_id"] = int(test_id[i])
        y_dict["score"] = float(y_pred[i])
        json.dump(y_dict, f)
        f.write("\n")
