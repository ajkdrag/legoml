from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from legoml.core.loops.trainer import Trainer
from legoml.core.strategies.single_device import SingleDeviceStrategy
from legoml.core.callbacks.progress import TQDMProgressBar
from legoml.core.callbacks.lr_monitor import LearningRateMonitor
from legoml.core.callbacks.checkpoint import ModelCheckpoint

from legoml.tasks.nlp_token_classification import NERTokenClassificationTask
from legoml.metrics.token_acc import TokenAccuracy
from legoml.nlp.hf_adapters import HFTokenClassificationModule
from legoml.data.nlp_collate import NERCollator
from legoml.utils.seed import worker_init_fn


def _build_ner_dataset(tokenizer, dataset_name: str = "conll2003", max_length: int = 128):
    try:
        from datasets import load_dataset
    except Exception as e:
        raise ImportError("Please install `datasets` to run NLP experiments.") from e

    raw = load_dataset(dataset_name)
    ner_feature = raw["train"].features["ner_tags"]
    label_list = ner_feature.feature.names  # type: ignore[attr-defined]

    def tokenize_and_align_labels(example):
        tokenized = tokenizer(
            example["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=max_length,
        )
        labels = []
        for i, word_ids in enumerate(tokenized.word_ids(batch_index=0) if hasattr(tokenized, 'word_ids') else []):
            pass  # placeholder to avoid linter error
        aligned_labels = []
        for words, ner_tags in zip(example["tokens"], example["ner_tags"]):
            tokenized_ex = tokenizer(
                words,
                truncation=True,
                is_split_into_words=True,
                max_length=max_length,
            )
            word_ids = tokenized_ex.word_ids()
            prev_word_idx = None
            labels_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    labels_ids.append(-100)
                elif word_idx != prev_word_idx:
                    labels_ids.append(ner_tags[word_idx])
                else:
                    labels_ids.append(-100)
                prev_word_idx = word_idx
            aligned_labels.append(labels_ids)

        tokenized = tokenizer(
            example["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=max_length,
        )
        tokenized["labels"] = aligned_labels
        return tokenized

    tokenized = raw.map(tokenize_and_align_labels, batched=True)
    cols = ["input_ids", "attention_mask", "labels"]
    tokenized = tokenized.remove_columns([c for c in tokenized["train"].column_names if c not in cols])
    tokenized.set_format(type="python")
    return tokenized, label_list


def finetune_tiny_bert_ner_v2(
    *,
    model_name: str = "prajjwal1/bert-tiny",
    dataset_name: str = "conll2003",
    max_epochs: int = 1,
    batch_size: int = 8,
    max_length: int = 128,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
    except Exception as e:
        raise ImportError("Please install `transformers` to run NLP experiments.") from e

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds, label_list = _build_ner_dataset(tokenizer, dataset_name=dataset_name, max_length=max_length)
    num_labels = len(label_list)

    hf_model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    model = HFTokenClassificationModule(hf_model)

    collator = NERCollator(tokenizer=tokenizer, label_pad_token_id=-100)
    train_loader = DataLoader(ds["train"], batch_size=batch_size, shuffle=True, collate_fn=collator, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(ds["validation"], batch_size=batch_size, shuffle=False, collate_fn=collator, worker_init_fn=worker_init_fn)

    task = NERTokenClassificationTask(
        model=model,
        metrics=[TokenAccuracy(ignore_index=-100)],
        device=device,
        ignore_index=-100,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = None

    trainer = Trainer(
        task=task,
        optimizer=optimizer,
        scheduler=scheduler,
        strategy=SingleDeviceStrategy(device=device, use_amp=False),
        callbacks=[
            TQDMProgressBar(leave=False),
            LearningRateMonitor(),
            ModelCheckpoint(save_dir="./models", run_name="bert_tiny_ner_v2", every_n_epochs=1),
        ],
        max_epochs=max_epochs,
        log_every_n_steps=50,
    )

    return trainer.fit(train_loader, val_loader)
