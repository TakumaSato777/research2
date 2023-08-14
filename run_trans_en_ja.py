from cmath import log
import os
import argparse
from pathlib import Path
import pathlib
from sre_constants import OP_IGNORE
from turtle import forward
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from transformers import BertForSequenceClassification, BertTokenizer, AdamW
import torchmetrics

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything

class ClassificationDataset(Dataset):
    def __init__(self, data_file: Path, tokenizer: AutoTokenizer) -> None:
        super().__init__()
        self.input_ids = []
        
        with open(data_file, 'r') as reader:
            for line in reader:
                row = line.strip().split(',')
                if int(row[0]) < 3:
                    l =  0
                else:
                    l = 1
                label = l
                text  = ' '.join(row[1:])

                encoded = tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length = 255,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )

                self.input_ids.append(
                    dict(
                        input_ids=encoded['input_ids'].flatten(),
                        attention_mask=encoded['attention_mask'].flatten(),
                        labels=torch.tensor(label)
                    )
                )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index) -> dict:
        return self.input_ids[index]


class ClassificationDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir: Path, batch_size : int, pretrained_model_name: str) -> None:
        super().__init__()
        self.train_file = dataset_dir.joinpath('trans_to_ja_train_100.csv')
        self.val_file = dataset_dir.joinpath('trans_to_ja_val_100.csv')
        self.test_file = dataset_dir.joinpath('ja_test.csv')
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading

    def setup(self, stage = None) -> None:
        self.train_dataset = ClassificationDataset(self.train_file, self.tokenizer)
        self.val_dataset = ClassificationDataset(self.val_file, self.tokenizer)
        self.test_dataset = ClassificationDataset(self.test_file, self.tokenizer)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class Classifier(pl.LightningModule):
    def __init__(self, roberta_model: AutoModelForSequenceClassification) -> None:
        super().__init__()
        self.model = roberta_model

        self.metric = torchmetrics.Accuracy()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs['loss'], outputs['logits']

    def training_step(self, batch, batch_idx):
        loss, preds = self.forward(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    labels=batch["labels"])
        return {'loss': loss,
                'batch_preds': preds,
                'batch_labels': batch["labels"]}

    def validation_step(self, batch, batch_idx):
        loss, preds = self.forward(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    labels=batch["labels"])
        return {'loss': loss,
                'batch_preds': preds,
                'batch_labels': batch["labels"]}

    def test_step(self, batch, batch_idx):
        loss, preds = self.forward(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    labels=batch["labels"])
        return {'loss': loss,
                'batch_preds': preds,
                'batch_labels': batch["labels"]}

    def training_epoch_end(self, outputs) -> None:
        epoch_loss = torch.sum(torch.tensor([x['loss'] for x in outputs]))
        self.log(
            'train_loss',
            epoch_loss,
            prog_bar=True,
            logger=True,
            on_epoch=True
        )

    def validation_epoch_end(self, outputs) -> None:
        epoch_loss = torch.sum(torch.tensor([x['loss'] for x in outputs]))
        self.log(
            'valid_loss',
            epoch_loss,
            prog_bar=True,
            logger=True,
            on_epoch=True
        )

    def test_epoch_end(self, outputs) -> None:
        epoch_loss = torch.sum(torch.tensor([x['loss'] for x in outputs]))
        epoch_preds = torch.cat([x['batch_preds'] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        self.metric(epoch_preds, epoch_labels)
        self.log(
            'test_loss',
            epoch_loss,
            prog_bar=True,
            logger=True,
            on_epoch=True
        )
        self.log(
            'test_accuracy',
            self.metric,
            prog_bar=True,
            logger=True,
            on_epoch=True
        )

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        scheduler = {'scheduler': StepLR(optimizer=optimizer, step_size=1, gamma=0.2)}
        return [optimizer]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=100, help='seed')
    args = parser.parse_args()

    seed=args.seed
    seed_everything(seed=seed, workers=True)
    ###change###
    num_epoch = 30

    pretrained_model_name = 'cyberagent/xlm-roberta-large-jnli-jsick'
    roberta_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=2)
    model = Classifier(roberta_model)

    # dataset_dir = pathlib.Path('dataset', 'flat-livedoor-news-corpus-20')
    dataset_dir = pathlib.Path('dataset', 'trans')
    ###change###
    batch_size = 16

    run_name = 'trans_en_ja_100'

    early_stop_callback = EarlyStopping(
        monitor='valid_loss', 
        min_delta=0.05, 
        patience=3, 
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints/{}-seed{}".format(run_name, seed),
        filename='{epoch}',
        verbose=True,
        monitor='valid_loss',
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=num_epoch,
        gpus=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=pl_loggers.TensorBoardLogger(save_dir='logs/{}/'.format(run_name))
    )

    data_module = ClassificationDataModule(dataset_dir=dataset_dir, batch_size=batch_size, pretrained_model_name=pretrained_model_name)

    #train_dataset = ClassificationDataset(data_module.train_file, data_module.tokenizer)
    #print(train_dataset[0])
    #print(train_dataset[1])

    trainer.fit(model=model, datamodule=data_module)

    result = trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=data_module)