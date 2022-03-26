from __future__ import print_function
from __future__ import print_function
import json
import json
import logging
import logging
import os
import os
import sys
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, RandomSampler, TensorDataset
import torch.utils.data.distributed
import torch_optimizer as optim
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import transformers

transformers.logging.set_verbosity_error()


PRE_TRAINED_MODEL_NAME = "Rostlab/prot_bert_bfd_localization"


class ProteinClassifier(nn.Module):
    def __init__(self, n_classes):
        super(ProteinClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.bert.config.hidden_size, n_classes),
            nn.Tanh(),
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(output.pooler_output)


# Network definition

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#logger.addHandler(logging.StreamHandler(sys.stdout))

MAX_LEN = None  # 512  # this is the max length of the sequence
PRE_TRAINED_MODEL_NAME = "Rostlab/prot_bert"
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=False)


def model_fn(model_dir):
    logger.info("model_fn")
    print("Loading the trained model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProteinClassifier(10)  # pass number of classes, in our case its 10
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f, map_location=device))
    return model.to(device)


def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
    if request_content_type == "application/json":
        sequence = json.loads(request_body)
        print("Input protein sequence: ", sequence)
        encoded_sequence = tokenizer.encode_plus(
            sequence,
            max_length=MAX_LEN,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded_sequence["input_ids"]
        attention_mask = encoded_sequence["attention_mask"]

        return input_ids, attention_mask

    raise ValueError("Unsupported content type: {}".format(request_content_type))


def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    input_id, input_mask = input_data
    logger.info(input_id, input_mask)
    input_id = input_id.to(device)
    input_mask = input_mask.to(device)
    with torch.no_grad():
        output = model(input_id, input_mask)
        _, prediction = torch.max(output, dim=1)
        return prediction


class ProteinSequenceDataset(Dataset):
    def __init__(self, sequence, targets, tokenizer, max_len):
        self.sequence = sequence
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, item):
        sequence1, sequence2 = self.sequence[item]
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            sequence1,
            sequence2,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        # print(encoding)
        # print(tokenizer.decode(encoding["input_ids"].flatten()))
        # exit()
        return {
            "protein_sequence": [sequence1, sequence2],
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long),
        }


# Network definition


iscuda = torch.cuda.is_available()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

MAX_LEN = 512  # this is the max length of the sequence
PRE_TRAINED_MODEL_NAME = "Rostlab/prot_bert_bfd_localization"
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=False)


def _get_train_data_loader(batch_size, data_x, data_y):
    train_data = ProteinSequenceDataset(
        sequence=data_x, targets=data_y, tokenizer=tokenizer, max_len=MAX_LEN
    )
    train_sampler = torch.utils.data.RandomSampler(
        train_data,
    )
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
    )
    return train_dataloader


def _get_test_data_loader(batch_size, data_x, data_y):
    test_data = ProteinSequenceDataset(
        sequence=data_x, targets=data_y, tokenizer=tokenizer, max_len=MAX_LEN
    )
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    return test_dataloader


def freeze(model, frozen_layers):
    modules = [model.bert.encoder.layer[:frozen_layers]]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False


def train(args):
    device = torch.device("cuda" if iscuda else "cpu")

    world_size = 1  # dist.get_world_size()
    rank = 0  # dist.get_rank()
    local_rank = 0  # dist.get_local_rank()

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if iscuda:
        torch.cuda.manual_seed(args.seed)

    train_loader = _get_train_data_loader(args.batch_size, args.data_x, args.data_y)
    if rank == 0:
        test_loader = _get_test_data_loader(args.batch_size, args.data_x, args.data_y)
        print("Max length of sequence: ", MAX_LEN)
        print("Freezing {} layers".format(args.frozen_layers))
        print("Model used: ", PRE_TRAINED_MODEL_NAME)

    logger.debug(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    model = ProteinClassifier(args.num_labels)  # The number of output labels.
    freeze(model, args.frozen_layers)
    # model = DDP(model.to(device), broadcast_buffers=False)
    if iscuda:
        torch.cuda.set_device(local_rank)
        model.cuda(local_rank)

    optimizer = optim.Lamb(
        model.parameters(),
        lr=args.lr * world_size,
        betas=(0.9, 0.999),
        eps=args.epsilon,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader.dataset)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for step, batch in enumerate(train_loader):
            b_input_ids = batch["input_ids"].to(device)
            b_input_mask = batch["attention_mask"].to(device)
            b_labels = batch["targets"].to(device)

            outputs = model(b_input_ids, attention_mask=b_input_mask)
            loss = loss_fn(outputs, b_labels)
            print(outputs, b_labels, loss)
            breakpoint()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            optimizer.zero_grad()

            if step % args.log_interval == 0 and rank == 0:
                logger.info(
                    "Collecting data from Master Node: \n Train Epoch: {} [{}/{} ({:.0f}%)] Training Loss: {:.6f}".format(
                        epoch,
                        step * len(batch["input_ids"]) * world_size,
                        len(train_loader.dataset),
                        100.0 * step / len(train_loader),
                        loss.item(),
                    )
                )
            if args.verbose:
                print("Batch", step, "from rank", rank)
        if rank == 0:
            test(model, test_loader, device)
        scheduler.step()
    if rank == 0:
        model_save = model.module if hasattr(model, "module") else model
        save_model(model_save, args.model_dir)


def save_model(model, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.state_dict(), path)
    logger.info(f"Saving model: {path} \n")


def test(model, test_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0
    loss_fn = nn.CrossEntropyLoss().to(device)
    tmp_eval_accuracy, eval_accuracy = 0, 0

    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch["input_ids"].to(device)
            b_input_mask = batch["attention_mask"].to(device)
            b_labels = batch["targets"].to(device)

            outputs = model(b_input_ids, attention_mask=b_input_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, b_labels)
            correct_predictions += torch.sum(preds == b_labels)
            losses.append(loss.item())

    print(
        "\nTest set: Validation loss: {:.4f}, Validation Accuracy: {:.0f}%\n".format(
            np.mean(losses),
            100.0 * correct_predictions.double() / len(test_loader.dataset),
        )
    )


class args:
    model_dir = "model"
    data_dir = "data"
    test = False
    num_gpus = int(iscuda)
    num_labels = 1

    batch_size = 4
    test_batch_size = 8
    epochs = 2
    lr = 0.3e-5
    weight_decay = 0.01
    seed = 43
    epsilon = 1e-8
    frozen_layers = 10
    verbose = False
    log_interval = 10


if __name__ == "__main__":
    train(args)
