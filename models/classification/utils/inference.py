import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def load_model_and_classes(model_path, class_mapping_path):
    """
    Загрузка модели и Json-а классов.

    Args:
        model_path: Путь к файлу модели (.pt)
        class_mapping_path: Путь к JSON файлу с классами

    Returns:
        Tuple containing:
            - PyTorch model
            - Dictionary mapping class indices to class names
            - Device (cuda/cpu)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    with open(class_mapping_path, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return model, idx_to_class, device


def prepare_image(image_path):
    """
    Подготовка изображения для подачи в модель.

    Args:
        image_path (str): Путь к файлу изображения

    Returns:
        tuple: Кортеж из двух элементов
            - image_tensor (torch.Tensor): Предобработанный тензор изображения
            - image (PIL.Image): Исходное изображение в формате PIL

    """
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor, image


def predict_image(model, image_tensor, device):
    """
    Получение предсказаний модели для изображения.

    Args:
        model (torch.nn.Module): PyTorch модель
        image_tensor (torch.Tensor): Предобработанный тензор изображения
        device (torch.device): Устройство для вычислений (CPU/GPU)

    Returns:
        numpy.ndarray: Массив вероятностей принадлежности к каждому классу

    Notes:
        - Модель должна быть в режиме eval()
        - Тензор изображения должен иметь размерность [1, C, H, W]
        - Возвращаемые вероятности нормализованы (сумма = 1)
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)

    probs = probabilities[0].cpu().numpy()
    return probs


def visualize_predictions(
    image, probs, idx_to_class, image_path, output_dir
):
    """
    Визуализация изображения и предсказаний модели с сохранением результата.

    Args:
        image (PIL.Image): Исходное изображение
        probs (numpy.ndarray): Массив вероятностей для каждого класса
        idx_to_class (dict): Словарь для преобразования индекса в название класса
        image_path (str): Путь к исходному изображению
        output_dir (str): Директория для сохранения результатов

    Returns:
        str: Путь к сохраненному файлу визуализации

    Notes:
        - Создает график с двумя частями: исходное изображение и график вероятностей
        - Сохраняет результат в формате PNG
        - Имя выходного файла содержит временную метку
    """
    # Создаем директорию, если её нет
    os.makedirs(output_dir, exist_ok=True)

    # Создаем график
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Отображаем исходное изображение
    ax1.imshow(image)
    ax1.axis("off")
    ax1.set_title("Input Image")

    # Создаем столбчатую диаграмму
    classes = list(idx_to_class.values())
    ax2.bar(classes, probs * 100)
    ax2.set_ylabel("Probability (%)")
    ax2.set_title("Class Probabilities")
    plt.xticks(rotation=45)

    # Добавляем подписи значений
    for i, v in enumerate(probs):
        ax2.text(
            i,
            v * 100,
            f"{v * 100:.1f}%",
            horizontalalignment="center",
            verticalalignment="bottom",
        )

    # Настраиваем layout
    plt.tight_layout()

    # Генерируем имя файла с той же временной меткой
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_image_name = os.path.splitext(os.path.basename(image_path))[0]
    plot_filename = f"prediction_{base_image_name}_{timestamp}_image.png"
    plot_path = os.path.join(output_dir, plot_filename)

    # Сохраняем график
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)

    # Откладка
    # plt.show()

    return plot_path


def save_predictions_to_json(
    image_path, probabilities, idx_to_class, output_dir
):
    """
    Сохранение результатов предсказаний в JSON файл.

    Args:
        image_path (str): Путь к исходному изображению
        probabilities (numpy.ndarray): Массив вероятностей для каждого класса
        idx_to_class (dict): Словарь для преобразования индекса в название класса
        output_dir (str): Директория для сохранения результатов

    Returns:
        str: Путь к сохраненному JSON файлу

    Notes:
        - Создает JSON с метаданными (время, пользователь, путь) и предсказаниями
        - Вероятности не ограничены в точности
        - Имя выходного файла содержит временную метку
    """
    # Создаем директорию для предсказаний
    os.makedirs(output_dir, exist_ok=True)

    # Формируем результаты предсказаний
    predictions = {
        idx_to_class[idx]: float(prob) * 100 for idx, prob in enumerate(probabilities)
    }

    # Создаем словарь с метаданными и результатами
    result = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": os.getenv("USER", "unknown"),
            "image_path": os.path.abspath(image_path),
        },
        "predictions": predictions,
    }

    # Генерируем имя файла на основе времени
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_image_name = os.path.splitext(os.path.basename(image_path))[0]
    json_filename = f"prediction_{base_image_name}_{timestamp}.json"
    json_path = os.path.join(output_dir, json_filename)

    # Сохраняем результаты в JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return json_path


def main():
    # Проверка кол-ва введенных аргументов
    if len(sys.argv) != 2:
        print("Используйте: python inference.py <path_to_image>")
        sys.exit(1)
    # Константы
    MODEL_PATH = "trained_model_classification.pt"
    CLASS_MAPPING_PATH = "class_to_idx.json"
    IMAGE_PATH = sys.argv[1]
    OUTPUT_DIR = "predictions"


    # Проверка существования файлов
    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: Файл модели не найден: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(CLASS_MAPPING_PATH):
        print(f"Ошибка: Файл маппинга классов не найден: {CLASS_MAPPING_PATH}")
        sys.exit(1)
    if not os.path.exists(IMAGE_PATH):
        print(f"Ошибка: Изображение не найдено: {IMAGE_PATH}")
        sys.exit(1)

    try:
        # Загрузка модели и классов
        model, idx_to_class, device = load_model_and_classes(
            MODEL_PATH, CLASS_MAPPING_PATH
        )

        # Подготовка изображения
        image_tensor, original_image = prepare_image(IMAGE_PATH)

        # Получение предсказаний
        probabilities = predict_image(model, image_tensor, device)

        # Вывод результатов в консоль
        print("\nPredictions:")
        for idx, prob in enumerate(probabilities):
            class_name = idx_to_class[idx]
            print(f"{class_name}: {prob * 100:.1f}%")

        # Визуализация и сохранение графика
        plot_path = visualize_predictions(
            original_image, probabilities, idx_to_class, IMAGE_PATH, output_dir=OUTPUT_DIR
        )
        print(f"Plot saved to: {plot_path}")

        # Сохранение результатов в JSON
        json_path = save_predictions_to_json(IMAGE_PATH, probabilities, idx_to_class, output_dir=OUTPUT_DIR)
        print(f"\nPredictions saved to: {json_path}")

    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем")
        sys.exit(0)

    except Exception as e:
        print(f"Ошибка: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
