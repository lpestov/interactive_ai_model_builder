import json
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import Resample
import numpy as np
import matplotlib.pyplot as plt

from panns_inference import AudioTagging

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# --- Константы из скрипта обучения ---
SR = 16000  # Целевая частота дискретизации
DURATION = 5  # Длительность в секундах
SAMPLES = SR * DURATION  # Количество сэмплов
PANNS_EMBEDDING_DIM = 2048 # Размерность эмбеддинга PANNs (Cnn14)

# --- Вспомогательная функция для подготовки аудио ---
def prepare_audio_for_inference(wav, sr, target_sr=SR, target_samples=SAMPLES):
    if wav.ndim > 1 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.ndim == 1:
        wav = wav.unsqueeze(0)

    if sr != target_sr:
        wav = Resample(sr, target_sr)(wav)

    current_samples = wav.size(1)
    if current_samples > target_samples:
        wav = wav[:, :target_samples]
    elif current_samples < target_samples:
        pad_size = target_samples - current_samples
        wav = nn.functional.pad(wav, (0, pad_size))
    return wav

def load_model_and_classes_sound(mlp_model_path, class_mapping_path, device):
    """
    Загрузка MLP-модели, Json-а классов и инициализация PANNs.

    Args:
        mlp_model_path: Путь к файлу с весами MLP-модели (.pt)
        class_mapping_path: Путь к JSON файлу с классами
        device: Устройство для вычислений

    Returns:
        Tuple containing:
            - mlp_model (torch.nn.Module): MLP модель
            - audio_tagger (AudioTagging): Инициализированная модель PANNs
            - idx_to_class (dict): Словарь отображения индексов на имена классов
    """
    # Инициализация PANNs AudioTagger для извлечения эмбеддингов
    print("Initializing PANNs AudioTagger...")
    audio_tagger = AudioTagging(checkpoint_path=None, device=str(device))
    print("PANNs AudioTagger initialized.")

    # Загрузка отображения классов
    with open(class_mapping_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

    # Определение архитектуры MLP-модели
    mlp_model = torch.load(mlp_model_path, map_location=device)
    mlp_model.to(device)
    mlp_model.eval()

    return mlp_model, audio_tagger, idx_to_class

def extract_audio_embedding(audio_path, audio_tagger_model, device):
    """
    Подготовка аудиофайла и извлечение PANNs эмбеддинга.

    Args:
        audio_path (str): Путь к аудиофайлу
        audio_tagger_model (AudioTagging): Инициализированная модель PANNs
        device (torch.device): Устройство для вычислений

    Returns:
        torch.Tensor: Эмбеддинг аудио (формат [1, embedding_dim])
        torch.Tensor: Предобработанная волновая форма аудио (для возможной визуализации)
    """
    try:
        wav, sr = torchaudio.load(audio_path)
    except Exception as e:
        raise ValueError(f"Error loading audio file {audio_path}: {e}")

    wav_prepared = prepare_audio_for_inference(wav, sr)
    wav_prepared_for_panns = wav_prepared.to(device)

    wav_np = wav_prepared_for_panns.squeeze(0).cpu().numpy()

    with torch.no_grad():
        _, embedding = audio_tagger_model.inference(wav_np[None, :])

    embedding_tensor = torch.from_numpy(embedding).to(device)

    return embedding_tensor, wav_prepared

def predict_sound(mlp_model, embedding_tensor, device):
    """
    Получение предсказаний MLP-модели для аудио эмбеддинга.

    Args:
        mlp_model (torch.nn.Module): MLP модель
        embedding_tensor (torch.Tensor): Эмбеддинг аудио [1, embedding_dim]
        device (torch.device): Устройство для вычислений

    Returns:
        numpy.ndarray: Массив вероятностей принадлежности к каждому классу
    """
    with torch.no_grad():
        embedding_tensor = embedding_tensor.to(device)
        outputs = mlp_model(embedding_tensor)
        probabilities = F.softmax(outputs, dim=1)

    probs_np = probabilities[0].cpu().numpy()
    return probs_np

def visualize_sound_predictions(audio_waveform, probs, idx_to_class, audio_path, output_dir):
    """
    Визуализация спектрограммы аудио и предсказаний модели с сохранением результата.

    Args:
        audio_waveform (torch.Tensor): Предобработанная волновая форма [1, SAMPLES]
        probs (numpy.ndarray): Массив вероятностей для каждого класса
        idx_to_class (dict): Словарь для преобразования индекса в название класса
        audio_path (str): Путь к исходному аудиофайлу
        output_dir (str): Директория для сохранения результатов

    Returns:
        str: Путь к сохраненному файлу визуализации
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))


    spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )
    mel_spectrogram = spec_transform(audio_waveform.cpu())
    db_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

    ax1.imshow(db_spectrogram.squeeze().numpy(), aspect='auto', origin='lower', cmap='viridis')
    ax1.set_title("Audio spectrogram")
    ax1.set_xlabel("Time (frames)")
    ax1.set_ylabel("Frequency (Mel bins)")


    classes = list(idx_to_class.values())
    if len(classes) != len(probs):
        print(f"Warning: number of classes ({len(classes)}) does not match number of probabilities ({len(probs)}). The plot may be incorrect.")


    bar_positions = np.arange(len(classes))
    ax2.bar(bar_positions, probs * 100, tick_label=classes)
    ax2.set_ylabel("Probability (%)")
    ax2.set_title("Class probabilities")
    plt.xticks(bar_positions, classes, rotation=45, ha="right")


    for i, v in enumerate(probs):
        ax2.text(
            i,
            v * 100 + 1,
            f"{v * 100:.1f}%",
            horizontalalignment="center",
            verticalalignment="bottom",
            )
    ax2.set_ylim(0, 110)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    plot_filename = f"prediction_{base_audio_name}_{timestamp}_sound.png"
    plot_path = os.path.join(output_dir, plot_filename)

    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return plot_path


def save_sound_predictions_to_json(audio_path, probabilities, idx_to_class, output_dir):
    """
    Сохранение результатов предсказаний для аудио в JSON файл.

    Args:
        audio_path (str): Путь к исходному аудиофайлу
        probabilities (numpy.ndarray): Массив вероятностей для каждого класса
        idx_to_class (dict): Словарь для преобразования индекса в название класса
        output_dir (str): Директория для сохранения результатов

    Returns:
        str: Путь к сохраненному JSON файлу
    """
    os.makedirs(output_dir, exist_ok=True)

    predictions = {
        idx_to_class[idx]: float(prob) * 100 for idx, prob in enumerate(probabilities)
    }

    result = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": os.getenv("USER", "unknown"),
            "audio_path": os.path.abspath(audio_path),
        },
        "predictions": predictions,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    json_filename = f"prediction_{base_audio_name}_{timestamp}_sound.json"
    json_path = os.path.join(output_dir, json_filename)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return json_path


def main_sound():
    if len(sys.argv) != 2:
        print("Usage: python inference_sound.py <path_to_audio_file>")
        sys.exit(1)

    MLP_MODEL_PATH = "utils/sound_classification/trained_model_sound_classification.pt"
    CLASS_MAPPING_PATH = "utils/sound_classification/class_to_idx.json"
    AUDIO_PATH = sys.argv[1]
    OUTPUT_DIR = "predictions/sounds"

    if not os.path.exists(MLP_MODEL_PATH):
        print(f"Error: MLP model file not found: {MLP_MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(CLASS_MAPPING_PATH):
        print(f"Error: Class mapping file not found: {CLASS_MAPPING_PATH}")
        sys.exit(1)
    if not os.path.exists(AUDIO_PATH):
        print(f"Error: Audio file not found: {AUDIO_PATH}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        mlp_model, audio_tagger, idx_to_class = load_model_and_classes_sound(
            MLP_MODEL_PATH, CLASS_MAPPING_PATH, device
        )

        audio_embedding, audio_waveform_prepared = extract_audio_embedding(AUDIO_PATH, audio_tagger, device)

        probabilities = predict_sound(mlp_model, audio_embedding, device)

        print("\nPredictions:")
        for idx, prob in enumerate(probabilities):
            class_name = idx_to_class[idx]
            print(f"  {class_name}: {prob * 100:.1f}%")

        plot_path = visualize_sound_predictions(
            audio_waveform_prepared,
            probabilities,
            idx_to_class,
            AUDIO_PATH,
            output_dir=OUTPUT_DIR,
        )
        print(f"Plot saved to: {plot_path}")

        json_path = save_sound_predictions_to_json(
            AUDIO_PATH, probabilities, idx_to_class, output_dir=OUTPUT_DIR
        )
        print(f"Predictions saved to JSON: {json_path}")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
        sys.exit(0)
    except Exception as e:
        import traceback
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main_sound()