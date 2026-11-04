import random, os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import json

from utils import init_logger, load_pair_data, cal_nwgm_center_attention
from data_load_utils import build_dataset
from models import Walk_with_bert
from torch.cuda.amp import autocast, GradScaler

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size.')
parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')

parser.add_argument("--evi_num", type=int, default=20, help='Evidence num.')
parser.add_argument("--max_seq_length", default=128, type=int,)
parser.add_argument("--overwrite", action="store_true", help="Run training even if results.json already exists.")

args = parser.parse_args()
if args.batch_size < 1 or args.epochs < 1 or args.evi_num < 1 or args.max_seq_length < 3:
    parser.error("batch_size, epochs, and evi_num must be positive; max_seq_length must be at least 3.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

dir_path = Path('outputs/causal_walk_politihop_3way/')

train_data_path = "data/politihop/original/politihop_train_adv.json"
dev_file_paths = [
    "data/politihop/original/politihop_valid_all.json",
    "data/politihop/original/politihop_test_all.json",
    "data/politihop/original/politihop_valid_adv.json",
    "data/politihop/original/politihop_test_adv.json",
    "data/politihop/original/politihop_valid_even.json",
    "data/politihop/original/politihop_test_even.json",
    "data/politihop/original/Hard_PolitiHop_by_gpt-3.5.json",
    "data/politihop/original/Hard_PolitiHop_by_gpt-4.0.json",
    ]
checkpoint_selection_path = dev_file_paths[0]
graph_rep_model_path = "pretrained_models/graph_rep_model_politihop.pt"
nclass = 3
max_length = 5
beam_size = 3
causal_method = "nwgm_attention"
n_cluster = 5
opt_steps = 1

dir_path.mkdir(parents=True, exist_ok=True)
if os.path.exists(dir_path / 'results.json') and not args.overwrite:
    print(dir_path / 'results already exists!')
    exit(0)
else:
    print(dir_path)
(dir_path / "models").mkdir(parents=True, exist_ok=True)
if args.overwrite:
    for output_file in (dir_path / "results.json", dir_path / "models" / "best.pt"):
        if output_file.exists():
            output_file.unlink()
logger = init_logger(dir_path / "log.log", overwrite=args.overwrite)

def correct_prediction(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct

def eval_model(model, dev_data_list, best_accuracy, accuracy, step, best_step, save_checkpoint=True):
    model.eval()
    current_accuracy = {}
    for path in dev_data_list:
        dev_data = dev_data_list[path]
        dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False)
        dev_tqdm_iterator = dev_dataloader
        correct_pred = 0.0
        running_loss = 0.0
        with torch.no_grad():
            for index, data in enumerate(dev_tqdm_iterator):
                data = [i.to(device) for i in data]
                input_ids, input_mask, segment_ids, labels, sent_labels, evi_labels = data

                outputs, outputs1, probability = model(data, train_centers, True)  # probability [batch,6,6]
                loss = F.nll_loss(torch.log(outputs.clamp_min(1e-12)), labels)
                loss1 = F.nll_loss(torch.log(outputs1.clamp_min(1e-12)), labels)
                loss = loss + loss1
                # loss = loss_func(outputs, labels)

                # mask = sent_labels == 0
                # mask = torch.zeros_like(mask,dtype=torch.float32).masked_fill(mask,float("-inf"))
                # evi_labels = evi_labels + mask
                # evi_labels = F.softmax(evi_labels,dim=-1)
                # evi_labels = evi_labels.unsqueeze(1) # [batch,1,6]

                # evi_labels = torch.log(1e-9 + evi_labels) # [batch,1,6]
                # probability = torch.log(1e-9 + probability) # [batch,6,6]

                # kl_loss = F.kl_div(probability, evi_labels, log_target=True)

                # loss = loss + 0.1 * kl_loss

                correct = correct_prediction(outputs, labels)
                correct_pred += correct
                running_loss += loss.item()
        
        dev_loss = running_loss / len(dev_dataloader)
        dev_accuracy = correct_pred / len(dev_data)
        logger.info('%s Dev total acc: %lf, total loss: %lf\r\n' % (path, dev_accuracy, dev_loss))

        accuracy[path].append(dev_accuracy.item())
        current_accuracy[path] = dev_accuracy.item()

        if save_checkpoint and path == checkpoint_selection_path and dev_accuracy > best_accuracy[path]:
            best_accuracy[path] = dev_accuracy.item()
            best_step[path] = step
            torch.save({'model': model.state_dict(),
                        'best_accuracy': best_accuracy[path],
                        'dev_losses': dev_loss,
                        'selection_dataset': path},
                       dir_path / "models" / "best.pt")
    model.train()
    return best_accuracy, accuracy, best_step, current_accuracy

logger.info("load data...")
train_data = build_dataset(train_data_path, args.evi_num, args.max_seq_length)
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
total_steps = len(train_dataloader) * args.epochs / opt_steps


feature_num = 768
dev_data_list = {}
best_accuracy = {checkpoint_selection_path: -1.0}
best_step = {checkpoint_selection_path: 0}
accuracy = {}
for path in dev_file_paths:
    dev_data = build_dataset(path, args.evi_num, args.max_seq_length)
    dev_data_list[path] = dev_data
    accuracy[path] = []
feature_num = 768
model = Walk_with_bert(feature_num, nclass, max_length, beam_size, args.evi_num, causal_method)
optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
model = model.to(device)

logger.info("start train...")
train_tqdm_iterator = tqdm(train_dataloader)
train_centers = cal_nwgm_center_attention(graph_rep_model_path, train_tqdm_iterator, nclass, device, feature_num, args.evi_num, n_cluster)
step = 0
for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    correct_pred = 0.0
    train_tqdm_iterator = tqdm(train_dataloader)
    for index, data in enumerate(train_tqdm_iterator):
        data = [i.to(device) for i in data]
        input_ids, input_mask, segment_ids, labels, sent_labels, evi_labels = data  # sent_labels [batch,6] evi_labels [batch,6]
    
        outputs, outputs1, probability = model(data, train_centers, True)  # probability [batch,6,6]
        loss = F.nll_loss(torch.log(outputs.clamp_min(1e-12)), labels)
        loss1 = F.nll_loss(torch.log(outputs1.clamp_min(1e-12)), labels)
        loss = loss + loss1
        # loss = loss_func(outputs, labels)

        # mask = sent_labels == 0
        # mask = torch.zeros_like(mask,dtype=torch.float32).masked_fill(mask,float("-inf"))
        # evi_labels = evi_labels + mask
        # evi_labels = F.softmax(evi_labels,dim=-1)
        # evi_labels = evi_labels.unsqueeze(1) # [batch,1,6]

        # evi_labels = torch.log(1e-9 + evi_labels) # [batch,1,6]
        # probability = torch.log(1e-9 + probability) # [batch,6,6]

        # kl_loss = F.kl_div(probability, evi_labels, log_target=True)

        # loss = loss + 0.1 * kl_loss

        correct = correct_prediction(outputs, labels)
        correct_pred += correct
        running_loss += loss.item()

        description = 'epoch %d Acc: %lf, Loss: %lf' % (epoch, correct_pred / (index + 1) / args.batch_size, running_loss / (index + 1))
        train_tqdm_iterator.set_description(description)

        loss = loss / opt_steps
        loss.backward()
        if index % opt_steps == 0 or index+1 == len(train_tqdm_iterator):
            optimizer.step()
            optimizer.zero_grad()
            # best_accuracy, accuracy, best_step = eval_model(model, dev_data_list, best_accuracy, accuracy, step, best_step)
            step += 1

    train_loss = running_loss / len(train_dataloader)
    train_accuracy = correct_pred / len(train_data)
    logger.info('epoch: %d, Train acc: %lf, total loss: %lf\r\n' % (epoch, train_accuracy, train_loss))
    best_accuracy, accuracy, best_step, _ = eval_model(
        model,
        {checkpoint_selection_path: dev_data_list[checkpoint_selection_path]},
        best_accuracy,
        accuracy,
        step,
        best_step,
    )

checkpoint = torch.load(dir_path / "models" / "best.pt", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model"])
_, _, _, final_accuracy = eval_model(
    model,
    dev_data_list,
    best_accuracy.copy(),
    {path: [] for path in dev_data_list},
    step,
    best_step.copy(),
    save_checkpoint=False,
)
logger.info(json.dumps(best_accuracy,indent=True))
logger.info(json.dumps(best_step,indent=True))
logger.info("total_steps: %d"%total_steps)

res = {
    "total_steps":total_steps,
    "best_accuracy":best_accuracy,
    "best_step":best_step,
    "final_accuracy":final_accuracy,
    "accuracy":accuracy,
}
with (dir_path / "results.json").open("w", encoding="utf-8") as fout:
    json.dump(res, fout, indent=2)