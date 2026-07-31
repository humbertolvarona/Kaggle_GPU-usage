# Complete Guide to Using Kaggle Notebooks with GPU Acceleration

This guide explains, step by step, how to run code in **Kaggle Notebooks**, also known as *Kernels*, using GPU acceleration. It covers account registration and verification, environment configuration, several deep learning examples, distributed multi-GPU training, efficient language-model fine-tuning, and remote execution through the Kaggle Public API.

## Table of Contents

1. [Account Registration and Phone Verification](#1-account-registration-and-phone-verification)
2. [Environment Setup and Preparation](#2-environment-setup-and-preparation)
3. [Fundamental GPU Examples](#3-fundamental-gpu-examples)
4. [Advanced Workflows](#4-advanced-workflows)
5. [Quotas, Best Practices, and the Public API](#5-quotas-best-practices-and-the-public-api)

---

## 1. Account Registration and Phone Verification

Access to Kaggle's free GPU hardware, such as an **NVIDIA Tesla P100** or two **NVIDIA T4 GPUs**, requires registering an account and completing the identity-verification process required by the platform.

### 1.1. Create a Kaggle Account

1. Go to [Kaggle](https://www.kaggle.com/).
2. Select **Register**.
3. Create an account using Google or an email address.

### 1.2. Verify Your Phone Number

1. Select your profile picture in the upper-right corner.
2. Open **Settings**.
3. Scroll to the **Phone Verification** section.
4. Enter the requested phone number.
5. Enter the verification code received by SMS.

> [!IMPORTANT]
> According to the original guide, GPU and TPU options may remain locked or produce runtime connection errors if phone verification is not completed.

---

## 2. Environment Setup and Preparation

### 2.1. Create a Notebook

1. Select **Code** in the left sidebar.
2. Select **New Notebook**.
3. Kaggle will open an interactive environment similar to Jupyter Notebook.

### 2.2. Enable GPU Acceleration

1. Open the **Settings** panel in the right sidebar.
2. Under **Accelerator**, select an available GPU option, such as:
   - `GPU P100`
   - `GPU T4 x2`
3. The session will restart automatically to provision a container with NVIDIA CUDA drivers.

### 2.3. Verify the Hardware

Run a verification cell at the beginning of the session to confirm that PyTorch and TensorFlow recognize the active GPU.

```python
import torch

print("PyTorch CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Count:", torch.cuda.device_count())
    print("Device Name:", torch.cuda.get_device_name(0))

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow GPU Available:", len(gpus) > 0)
if gpus:
    for gpu in gpus:
        print("Detected GPU:", gpu)
```

### 2.4. Install Dependencies

Kaggle includes many preinstalled data-science and deep-learning frameworks and packages, including:

- PyTorch
- TensorFlow
- Scikit-Learn
- Pandas
- Ultralytics YOLO
- Hugging Face Transformers

To install or update specific packages, run a shell command inside a notebook cell:

```bash
!pip install -q -U ultralytics transformers peft accelerate bitsandbytes
```

### 2.5. Manage Inputs and Outputs

#### Input Data

Datasets are added through **Add Input** in the right panel. They may come from competitions, public datasets, or custom uploads.

Kaggle mounts these data under:

```text
/kaggle/input/
```

This directory is **read-only**.

#### Output Files

Generated files, checkpoints, and exported models should be saved under:

```text
/kaggle/working/
```

This directory supports both read and write operations. Results can be downloaded after execution.

---

## 3. Fundamental GPU Examples

### 3.1. Image Classification with PyTorch

This example uses transfer learning with a pretrained **ResNet-18** network. The classification layer is replaced to support ten classes, and both the model and tensors are moved to the GPU.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Executing on device:", device)

x_dummy = torch.randn(64, 3, 224, 224)
y_dummy = torch.randint(0, 10, (64,))
dataset = TensorDataset(x_dummy, y_dummy)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(3):
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/3] - Mean Loss: {total_loss / len(loader):.4f}")
```

### 3.2. Object Detection with YOLOv8

Modern computer-vision libraries can manage GPU selection through a parameter such as `device=0`, which represents the first available CUDA GPU.

```python
import torch
from ultralytics import YOLO

print("GPU available for YOLO:", torch.cuda.is_available())

model = YOLO("yolov8n.pt")

results = model.train(
    data="coco8.yaml",
    epochs=5,
    imgsz=640,
    batch=4,
    device=0,
    project="/kaggle/working/yolo_experiments",
    name="run_demo",
)

print("Training finished! Results saved to /kaggle/working/yolo_experiments")
```

The `coco8.yaml` demonstration dataset is downloaded automatically.

### 3.3. Text Classification with Hugging Face Transformers

This example performs accelerated inference to classify the sentiment of several texts. The model and the input tensor dictionary are moved to the CUDA device.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)

texts = [
    "Kaggle notebooks with free GPUs make deep learning so fast and enjoyable!",
    "I am having trouble configuring my environment and it is frustrating.",
]

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
inputs = {key: val.to(device) for key, val in inputs.items()}

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

labels = ["Negative", "Positive"]
for text, pred in zip(texts, predictions):
    print(f"Text: '{text[:45]}...' -> Sentiment: {labels[pred.item()]}")
```

---

## 4. Advanced Workflows

### 4.1. Distributed Training with PyTorch DDP

When Kaggle provides two T4 GPUs, conventional PyTorch code normally uses only `cuda:0`. To train with both GPUs simultaneously, use **Distributed Data Parallel**, creating multiple processes through `torch.multiprocessing`.

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.net(x)


def train_worker(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    data = torch.randn(128, 3, 64, 64)
    targets = torch.randint(0, 10, (128,))
    dataset = TensorDataset(data, targets)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    loader = DataLoader(dataset, batch_size=16, sampler=sampler)

    model = SimpleConvNet().to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(2):
        sampler.set_epoch(epoch)
        total_loss = 0.0

        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(rank), y_batch.to(rank)
            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if rank == 0:
            print(
                f"[Rank 0] Epoch {epoch + 1} "
                f"- Mean Loss: {total_loss / len(loader):.4f}"
            )

    cleanup()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print(f"Spawning DDP across {n_gpus} GPU(s)...")

    if n_gpus > 1:
        mp.spawn(
            train_worker,
            args=(n_gpus,),
            nprocs=n_gpus,
            join=True,
        )
    else:
        print("Multi-GPU not detected. Make sure 'GPU T4 x2' is selected in Settings.")
```

### 4.2. Parameter-Efficient Fine-Tuning with PEFT and QLoRA

Fine-tuning all the weights of a language model can exceed the available VRAM. The guide proposes combining:

- 4-bit quantization through `bitsandbytes`.
- QLoRA with the `nf4` quantization type.
- Low-rank adapters through `peft`.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model_id = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=2,
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

raw_data = {
    "text": [
        "GPU acceleration makes model training incredibly fast!",
        "I experienced an Out-Of-Memory error while loading the full weights.",
    ]
    * 8,
    "label": [1, 0] * 8,
}

dataset = Dataset.from_dict(raw_data)
tokenized_ds = dataset.map(
    lambda x: tokenizer(
        x["text"],
        padding="max_length",
        truncation=True,
        max_length=32,
    ),
    batched=True,
)
tokenized_ds.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"],
)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
model.train()

for epoch in range(2):
    for idx in range(0, len(tokenized_ds), 4):
        batch = tokenized_ds[idx : idx + 4]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(
        f"LoRA Fine-Tuning Epoch {epoch + 1} Completed "
        f"- Final Batch Loss: {loss.item():.4f}"
    )
```

### 4.3. Custom Dataset and Automatic Mixed Precision

To increase throughput and reduce out-of-memory errors, the guide combines:

- `pin_memory=True` in the `DataLoader`.
- Transfers with `non_blocking=True`.
- Automatic mixed precision through `autocast`.
- Gradient scaling through `GradScaler`.

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SyntheticImageDataset(Dataset):
    def __init__(self, n_samples=300, img_size=224):
        self.n_samples = n_samples
        self.img_size = img_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        image = torch.randn(3, self.img_size, self.img_size)
        label = torch.tensor(1 if idx % 2 == 0 else 0, dtype=torch.long)
        return image, label


dataset = SyntheticImageDataset()
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(64, 2),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = GradScaler()

model.train()
for epoch in range(3):
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    print(
        f"AMP Epoch [{epoch + 1}/3] "
        f"- Mean Loss: {total_loss / len(loader):.4f}"
    )
```

### 4.4. Batch Inference and VRAM Management

To process large test datasets, the guide recommends disabling the automatic-differentiation graph with `torch.no_grad()` and periodically performing garbage collection and clearing the CUDA cache.

```python
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_data = torch.randn(10000, 3, 128, 128)
test_dataset = TensorDataset(test_data)
inference_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(16, 5),
).to(device)
model.eval()

all_predictions = []

print("Starting high-throughput batch inference...")
with torch.no_grad():
    for batch_idx, (x_batch,) in enumerate(inference_loader):
        x_batch = x_batch.to(device)
        logits = model(x_batch)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_predictions.extend(preds)

        if (batch_idx + 1) % 25 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            print(
                f"Batch {batch_idx + 1}/{len(inference_loader)} completed "
                "- VRAM Cache cleared."
            )

print(
    f"Inference complete! Successfully predicted "
    f"{len(all_predictions)} samples."
)
```

---

## 5. Quotas, Best Practices, and the Public API

### 5.1. Quotas and Runtime Limits

The original guide provides the following values:

- 30 free GPU hours per week.
- Weekly quota reset at midnight Central European Time on Saturday.
- Maximum duration of nine hours per interactive session or saved run.
- Usage monitoring under **Profile > Settings > Quotas**.

The guide also states that linking an active Google Colab Pro or Colab Pro+ subscription may add GPU hours to the weekly quota. This statement is retained as part of the original content.

### 5.2. Best Practices for Conserving GPU Quota

Do not use the GPU for conventional CPU workloads. Typical Pandas, NumPy, and Scikit-Learn operations do not directly use the GPU.

Keep the accelerator disabled during:

- Initial exploratory data analysis.
- Data cleaning.
- Conventional CPU-based preprocessing.

Enable the GPU when deep-learning model training or inference begins.

### 5.3. Stop Interactive Sessions

Closing the browser window may leave the session running until the idle timeout is reached. To stop it manually:

1. Open **Active Events** in the lower-left corner.
2. Locate the running session.
3. Select **Stop**.

### 5.4. Save Version Options

#### Save & Run All

Runs the entire notebook from beginning to end on the server. This option is appropriate for generating final files such as `submission.csv`.

#### Quick Save

Immediately saves a copy of the code without rerunning every cell or consuming a complete GPU run.

### 5.5. Remote Execution Through the Kaggle Public API

The API allows you to manage and submit notebooks from a local terminal.

#### Install the Package

```bash
pip install kaggle
```

#### Download the Token

1. Open **Profile > Settings**.
2. Select **Create New Token**.
3. Download the `kaggle.json` file.
4. Place it in:

```text
~/.kaggle/kaggle.json
```

#### Enable the GPU in the Metadata

The notebook directory must include a `kernel-metadata.json` file with the following property:

```json
{
  "enable_gpu": true
}
```

#### Push the Notebook

```bash
kaggle kernels push -p /path/to/notebook_folder
```

---

## Summary

The general workflow is as follows:

1. Register and verify the Kaggle account.
2. Create a new notebook.
3. Select a GPU under **Settings > Accelerator**.
4. Verify CUDA availability.
5. Add data under `/kaggle/input/`.
6. Save results under `/kaggle/working/`.
7. Use a single GPU, multiple GPUs, mixed precision, or quantization according to the workflow.
8. Stop the session manually when finished.
9. Save a complete version or submit the notebook through the Public API.
