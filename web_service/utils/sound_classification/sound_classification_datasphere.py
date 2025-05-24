# -- Данный код предназначен для загрузки модели в Yandex DataSphere и не ориентирован для дальнейших экспериментов --

import json
import os
import sys
import zipfile
import random
import logging
import contextlib
import io

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
from torchaudio.transforms import Resample
from panns_inference import AudioTagging

if __name__ == "__main__":
    SR = 16000  # Целевая частота дискретизации
    DURATION = 5  # Длительность в секундах
    SAMPLES = SR * DURATION  # Количество сэмплов


    # --- Классы аугментации волновой формы ---
    class GaussianNoise(nn.Module):
        def __init__(self, snr_db=15, p=0.5):
            super().__init__()
            self.snr_db = snr_db
            self.p = p

        def forward(self, waveform):
            if torch.rand(1) < self.p:
                signal_power = waveform.norm(p=2)
                noise = torch.randn_like(waveform)
                noise_power = noise.norm(p=2)
                if noise_power == 0:
                    return waveform
                snr = 10 ** (self.snr_db / 10)
                scale = (signal_power / (noise_power * snr)).sqrt()
                return waveform + scale * noise
            return waveform


    class RandomVolume(nn.Module):
        def __init__(self, min_gain_db=-6, max_gain_db=6, p=0.5):
            super().__init__()
            self.min_gain_db = min_gain_db
            self.max_gain_db = max_gain_db
            self.p = p

        def forward(self, waveform):
            if random.random() < self.p:
                gain_db = random.uniform(self.min_gain_db, self.max_gain_db)
                gain_factor = 10 ** (gain_db / 20)
                gain_factor = max(gain_factor, 1e-5)
                return waveform * gain_factor
            return waveform


    # --- Вспомогательная функция для подготовки аудио ---
    def prepare_audio(wav, sr, target_sr=SR, target_samples=SAMPLES):
        if wav.ndim > 1 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        elif wav.ndim == 1:
            wav = wav.unsqueeze(0)

        if sr != target_sr:
            wav = Resample(sr, target_sr)(wav)

        if wav.size(1) > target_samples:
            wav = wav[:, :target_samples]
        else:
            pad_size = target_samples - wav.size(1)
            wav = nn.functional.pad(wav, (0, pad_size))
        return wav


    # --- Класс Dataset для эмбеддингов PANNs ---
    class PANNsEmbedDatasetDS(Dataset):
        def __init__(self, original_folder, class_to_idx_map, audio_tagger_model,
                     waveform_augment_transform=None, target_size_per_class=50):
            self.original_folder = original_folder
            self.class_to_idx = class_to_idx_map
            self.audio_tagger = audio_tagger_model
            self.waveform_augment = waveform_augment_transform
            self.target_size_per_class = target_size_per_class
            self.samples = []

            # Проверка на отсутствующие классы
            class_names_in_folder = os.listdir(original_folder)
            missing_classes = [
                cls for cls in self.class_to_idx.keys() if cls not in class_names_in_folder
            ]
            if missing_classes:
                raise ValueError(
                    f"Классы {missing_classes} из class_to_idx.json отсутствуют в папке {original_folder}"
                )

            for class_name in class_names_in_folder:
                if class_name not in self.class_to_idx:
                    print(f"Warning: Folder {class_name} not found in class_to_idx.json, skipping.")
                    continue

                label = self.class_to_idx[class_name]
                class_path = os.path.join(original_folder, class_name)
                if not os.path.isdir(class_path):
                    continue

                audio_files = [os.path.join(class_path, f) for f in os.listdir(class_path)
                               if f.lower().endswith(('.wav', '.mp3', '.flac'))]

                if not audio_files:
                    print(f"Warning: Audio files not found in {class_path} for class {class_name}")
                    continue

                for i in range(self.target_size_per_class):
                    file_path = audio_files[i % len(audio_files)]
                    self.samples.append((file_path, label))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            file_path, label = self.samples[idx]
            try:
                wav, sr = torchaudio.load(file_path)
            except Exception as e:
                print(f"Error loading audio file {file_path}: {e}")
                dummy_embedding = torch.zeros(2048)
                return dummy_embedding, -1

            wav_prepared = prepare_audio(wav, sr)

            if self.waveform_augment:
                wav_prepared = self.waveform_augment(wav_prepared)

            wav_np = wav_prepared.squeeze(0).cpu().numpy()

            with torch.no_grad():
                _, embedding = self.audio_tagger.inference(wav_np[None, :])

            embedding = torch.from_numpy(embedding).squeeze(0)

            return embedding, label

    class suppress_stdout_stderr(object):
        def __init__(self):
            self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
            self.save_fds = [os.dup(1), os.dup(2)]

        def __enter__(self):
            os.dup2(self.null_fds[0], 1)
            os.dup2(self.null_fds[1], 2)

        def __exit__(self, *_):
            os.dup2(self.save_fds[0], 1)
            os.dup2(self.save_fds[1], 2)
            for fd in self.null_fds + self.save_fds:
                os.close(fd)

    print("Starting sound classification training for DataSphere")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Загрузка гиперпараметров
    with open("hyperparams.json", "r") as f:
        hyperparams = json.load(f)

    # Загрузка отображения классов на индексы
    with open("class_to_idx.json", "r") as f:
        class_to_idx = json.load(f)

    num_classes = len(class_to_idx)
    print(f"Number of classes: {num_classes}")
    print(f"Class mapping: {class_to_idx}")

    dataset_zip_name = "sound_dataset.zip"
    dataset_extract_folder = "sound_dataset_extracted"
    train_folder_path = os.path.join(dataset_extract_folder, "sound_dataset/train")

    if os.path.exists(dataset_extract_folder):
        import shutil

        shutil.rmtree(dataset_extract_folder)
        print(f"Removed existing directory: {dataset_extract_folder}")

    print(f"Unpacking {dataset_zip_name}...")
    with zipfile.ZipFile(dataset_zip_name, "r") as zip_ref:
        zip_ref.extractall(dataset_extract_folder)
    print("Unpacking completed.")

    if not os.path.exists(train_folder_path):
        potential_base_list = os.listdir(dataset_extract_folder)
        if not potential_base_list:
            raise FileNotFoundError(f"Folder {dataset_extract_folder} is empty after unpacking.")
        potential_base = os.path.join(dataset_extract_folder, potential_base_list[0])
        if os.path.isdir(potential_base) and "train" in os.listdir(potential_base):
            train_folder_path = os.path.join(potential_base, "train")
        else:
            raise FileNotFoundError(
                f"Train folder not found at path {train_folder_path} or in expected subdirectories.")
    print(f"Using training data from: {train_folder_path}")

    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("torchaudio").setLevel(logging.ERROR)

    print("Initializing PANNs AudioTagger...")
    ## Delete PANNs logs
    with suppress_stdout_stderr():
        audio_tagger = AudioTagging(checkpoint_path=None, device=str(DEVICE))
        print("PANNs AudioTagger initialized.")
    panns_embedding_dim = 2048

    # Определение аугментаций волновой формы
    waveform_augment_transform = nn.Sequential(
        RandomVolume(min_gain_db=-6, max_gain_db=6, p=0.5),
        GaussianNoise(snr_db=30, p=0.5),
    ).to(DEVICE)

    # Создание Dataset и DataLoader
    print("Creating dataset...")
    train_dataset = PANNsEmbedDatasetDS(
        original_folder=train_folder_path,
        class_to_idx_map=class_to_idx,
        audio_tagger_model=audio_tagger,
        waveform_augment_transform=waveform_augment_transform,
        target_size_per_class=hyperparams.get("target_size", 50)  # из hyperparams или по умолчанию
    )
    print(f"Dataset created with {len(train_dataset)} samples.")

    if len(train_dataset) == 0:
        print("Error: Dataset is empty. Check paths and data.")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparams["batch_size"],
        shuffle=True
    )

    # Определение MLP-модели
    mlp_model = nn.Sequential(
        nn.Linear(panns_embedding_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    ).to(DEVICE)

    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        mlp_model.parameters(),
        lr=hyperparams["learning_rate"],
        weight_decay=hyperparams.get("weight_decay", 0.0)
    )

    # Цикл обучения
    print("Starting training...")
    for epoch in range(hyperparams["num_epochs"]):
        mlp_model.train()
        epoch_train_loss = 0.0

        for embeddings, labels in train_loader:
            if embeddings.nelement() == 0:
                print("Skipping empty batch.")
                continue
            if -1 in labels:
                print("Skipping batch with data loading error.")
                continue

            embeddings, labels = embeddings.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = mlp_model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * embeddings.size(0)

        avg_epoch_train_loss = epoch_train_loss / len(train_loader.dataset)

        # Валидация на обучающем наборе
        mlp_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for embeddings, labels in train_loader:
                if embeddings.nelement() == 0 or -1 in labels: continue
                embeddings, labels = embeddings.to(DEVICE), labels.to(DEVICE)
                outputs = mlp_model(embeddings)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0

        print(f"Epoch {epoch + 1}/{hyperparams['num_epochs']} - "
              f"Loss: {avg_epoch_train_loss:.4f}, "
              f"Accuracy (On TRAIN Dataset): {accuracy:.4f}")

    # Сохранение обученной MLP-модели
    model_save_path = "trained_model_sound_classification.pt"
    torch.save(mlp_model, model_save_path)
    print(f"MLP-model saved in {model_save_path}")

    print("Studing has been ended.")
    sys.exit(0)
