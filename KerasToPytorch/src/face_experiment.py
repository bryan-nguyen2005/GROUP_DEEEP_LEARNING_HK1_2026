"""Thí nghiệm Bài 3 và 4. Chạy bằng python -m src.face_experiment --help."""
import argparse
import csv
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.models.gender_cnn import GenderCNN
from src.models.emotion_cnn import EmotionCNN
from src.pytorch_core import EarlyStopping, evaluate, fit, get_device, seed_everything
from src.mnist_experiment import use_glorot_initialization

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
GENDERS = ['male', 'female']  # UTKFace: 0 male, 1 female
REFERENCE = {'gender': {'baseline': (.4602, .8264), 'callbacks': (.3237, .8773)},
             'emotion': {'baseline': (1.4070, .5954), 'augmentation': (1.0158, .6163)}}


class FaceDataset(Dataset):
    def __init__(self, records, transform):
        self.records, self.transform = records, transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        source, label = self.records[index]
        if isinstance(source, Path):
            with Image.open(source) as im:
                image = im.convert('RGB')
        else:
            image = Image.fromarray(source, mode='L')
        return self.transform(image), label


def gender_records(root, seed=42):
    root = Path(root)
    records = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        parts = path.name.split('_')
        if len(parts) >= 4 and parts[0].isdigit() and parts[1] in ('0', '1'):
            records.append((path, int(parts[1])))
    if len(records) < 10 or len({label for _, label in records}) != 2:
        raise ValueError(f'Không tìm thấy đủ ảnh UTKFace có hai nhãn trong {root}; xem data/README_DATA.md')
    # Stratified deterministic split, one image used in exactly one set.
    splits = [[], [], []]
    rng = random.Random(seed)
    for label in (0, 1):
        group = [r for r in records if r[1] == label]
        if len(group) < 3:
            raise ValueError('Mỗi lớp UTKFace phải có ít nhất 3 ảnh')
        rng.shuffle(group)
        n = len(group)
        a = min(n - 2, max(1, int(.8 * n)))
        b = min(n - 1, max(a + 1, int(.9 * n)))
        for dest, part in zip(splits, (group[:a], group[a:b], group[b:])):
            dest.extend(part)
    return splits


def emotion_records(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Không thấy {path}; xem data/README_DATA.md')
    splits = {'Training': [], 'PublicTest': [], 'PrivateTest': []}
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'emotion', 'pixels', 'Usage'}.issubset(reader.fieldnames or []):
            raise ValueError('CSV phải có cột emotion, pixels, Usage')
        for line, row in enumerate(reader, 2):
            label = int(row['emotion'])
            values = np.fromstring(row['pixels'], dtype=np.int32, sep=' ')
            if row['Usage'] not in splits or not 0 <= label < 7 or values.size != 2304 or np.any((values < 0) | (values > 255)):
                raise ValueError(f'Dòng CSV không hợp lệ: {line}')
            splits[row['Usage']].append((values.astype(np.uint8).reshape(48, 48), label))
    if any(not value for value in splits.values()):
        raise ValueError('CSV phải có Training, PublicTest và PrivateTest')
    return list(splits.values())


def make_loaders(task, path, variant, batch_size, seed):
    if task == 'gender':
        records = gender_records(path, seed)
        base = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
        train_transform = base
    else:
        records = emotion_records(path)
        base = transforms.ToTensor()
        train_transform = (transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.RandomAffine(degrees=10, translate=(.1, .1), scale=(.9, 1.1)),
            transforms.ToTensor()]) if variant == 'augmentation' else base)
    datasets = [FaceDataset(rows, train_transform if i == 0 else base) for i, rows in enumerate(records)]
    loaders = [DataLoader(ds, batch_size=batch_size, shuffle=(i == 0),
                          generator=torch.Generator().manual_seed(seed), num_workers=0)
               for i, ds in enumerate(datasets)]
    return loaders


def run(task, path, variant, *, epochs=30, batch_size=128, lr=.001, seed=42):
    valid = {'gender': {'baseline', 'callbacks'}, 'emotion': {'baseline', 'augmentation'}}
    if variant not in valid[task]:
        raise ValueError(f'Biến thể không hợp lệ: {task}/{variant}')
    seed_everything(seed)
    device = get_device()
    train, val, test = make_loaders(task, path, variant, batch_size, seed)
    model = (GenderCNN() if task == 'gender' else EmotionCNN()).to(device)
    model.apply(use_glorot_initialization)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early = EarlyStopping(patience=7) if variant == 'callbacks' else None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.1, patience=3) if early else None
    prefix = '03_gender' if task == 'gender' else '04_emotion'
    for folder in ('logs', 'figures', 'checkpoints'):
        Path(folder).mkdir(exist_ok=True)
    start = time.time()
    history = fit(model, train, val, criterion, optimizer, device, epochs=epochs,
                  early_stopping=early, scheduler=scheduler,
                  csv_path=f'logs/history_{prefix}_{variant}.csv')
    seconds = time.time() - start
    metrics = evaluate(model, test, criterion, device)
    reference_loss, reference_accuracy = REFERENCE[task][variant]
    result = dict(task=task, variant=variant, train_count=len(train.dataset), val_count=len(val.dataset),
                  test_count=len(test.dataset), seed=seed, epochs_run=len(history), batch_size=batch_size,
                  initial_lr=lr, optimizer='Adam', parameters=sum(p.numel() for p in model.parameters()),
                  test_loss=metrics['loss'], test_accuracy=metrics['accuracy'], keras_reference_loss=reference_loss,
                  keras_reference_accuracy=reference_accuracy, seconds=seconds, device=str(device), torch_version=torch.__version__)
    with open(f'logs/result_{prefix}_{variant}.csv', 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=result.keys()); writer.writeheader(); writer.writerow(result)
    torch.save({'state_dict': model.state_dict(), 'task': task, 'variant': variant, 'seed': seed},
               f'checkpoints/{prefix}_{variant}.pt')
    print('Test:', result)
    return result


def plot_comparison(task):
    prefix, variants = ('03_gender', ['baseline', 'callbacks']) if task == 'gender' else ('04_emotion', ['baseline', 'augmentation'])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout='constrained')
    for variant in variants:
        with open(f'logs/history_{prefix}_{variant}.csv', encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        x = [int(r['epoch']) for r in rows]
        for ax, metric in zip(axes[:2], ('loss', 'accuracy')):
            for split, style in [('train', '-'), ('val', '--')]:
                ax.plot(x, [float(r[f'{split}_{metric}']) for r in rows], style, label=f'{variant} {split}')
        axes[2].plot(x, [float(r['lr']) for r in rows], label=variant)
    for ax, title in zip(axes, ['Cross-entropy', 'Accuracy', 'Learning rate']):
        ax.set(xlabel='Epoch', ylabel=title); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.savefig(f'figures/{prefix}_before_after_{"callbacks" if task == "gender" else "aug"}.png', dpi=300)
    plt.close(fig)


def demo_gender(path, batch_size=8, seed=42):
    _, _, test = make_loaders('gender', path, 'callbacks', batch_size, seed)
    device = get_device(); model = GenderCNN().to(device)
    checkpoint = torch.load('checkpoints/03_gender_callbacks.pt', map_location=device, weights_only=True)
    if checkpoint['seed'] != seed:
        raise ValueError('Seed demo phải khớp seed của checkpoint')
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    # Demo dùng các ảnh người lớn trong test, không dùng các ảnh tập train.
    adult_records = [r for r in test.dataset.records if int(r[0].name.split('_')[0]) >= 18]
    if not adult_records:
        raise ValueError('Không có ảnh người lớn trong tập test để minh họa')
    demo = DataLoader(FaceDataset(adult_records, test.dataset.transform), batch_size=batch_size)
    images, labels = next(iter(demo))
    with torch.no_grad():
        probs = model(images.to(device)).softmax(dim=1).cpu()
    fig, axes = plt.subplots(1, len(images), figsize=(2*len(images), 2.8), squeeze=False)
    for ax, im, label, prob in zip(axes[0], images, labels, probs):
        pred = int(prob.argmax()); ax.imshow(im.permute(1, 2, 0)); ax.axis('off')
        ax.set_title(f'Thật: {GENDERS[label]}\n{GENDERS[pred]}: {prob[pred]:.1%}', fontsize=8)
    fig.tight_layout(); fig.savefig('figures/03_gender_demo_predictions.png', dpi=300); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['gender', 'emotion'])
    parser.add_argument('--data', required=True, help='Thư mục UTKFace hoặc đường dẫn fer2013.csv')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    variants = ['baseline', 'callbacks'] if args.task == 'gender' else ['baseline', 'augmentation']
    for variant in variants:
        run(args.task, args.data, variant, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    plot_comparison(args.task)
    if args.task == 'gender':
        demo_gender(args.data, seed=args.seed)


if __name__ == '__main__':
    main()
