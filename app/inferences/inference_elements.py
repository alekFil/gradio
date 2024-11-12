import random

import numpy as np
import torch
import torch.nn.functional as F
from inferences.models.elements.elements_model import BiLSTMModel
from torch.nn.utils.rnn import pad_sequence


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(50)


class ModelElementsInference:
    def __init__(self, model_path, parameters, num_classes):
        # Инициализация модели и загрузка состояния
        self.parameters = parameters
        self.num_classes = num_classes
        self.model = BiLSTMModel(
            input_dim=self.parameters[3],
            num_classes=num_classes,
            hidden_dim=self.parameters[0],
            num_layers=self.parameters[1],
            dropout=self.parameters[2],
            use_attention=self.parameters[6],
        )
        self.model.load_state_dict(
            torch.load(
                model_path,
                weights_only=False,
                map_location=torch.device("cpu"),
            )
        )
        self.model.eval()  # Переключение модели в режим инференса

    def predict(self, features, lengths, mask):
        # Метод для выполнения предсказания
        with torch.no_grad():
            outputs, hidden_state = self.model(
                features=features,
                lengths=lengths,
                hidden_state=None,
                mask=mask,
            )

        valid_sequences_mask = mask.sum(dim=1) > 0
        outputs_filtered = outputs[valid_sequences_mask]
        mask_filtered = mask[valid_sequences_mask]

        outputs_masked = outputs_filtered.view(-1, self.num_classes)[
            mask_filtered.view(-1)
        ]

        # Для задачи классификации извлекаем максимальный логит
        predicted_probs = F.softmax(outputs_masked, dim=-1)
        predicted_labels = outputs_masked.argmax(dim=-1)

        return predicted_labels, predicted_probs


# Загрузка модели при старте сервиса
model_path = "app/inferences/models/elements/checkpoints/checkpoint.pt"
parameters = (64, 2, 0.3, 198, 0.05, 128, True, True, True, 0.05)
num_classes = 3
INFERENCE_ELEMENTS = ModelElementsInference(model_path, parameters, num_classes)


def predict(landmarks_data, world_landmarks_data):
    landmarks_tensor = torch.tensor(landmarks_data)
    world_landmarks_tensor = torch.tensor(world_landmarks_data)
    print(f"{landmarks_tensor.shape=}")
    print(f"{world_landmarks_tensor.shape=}")

    def collate_ml(batch):
        (
            lengths,
            swfeatures,
            sfeatures,
        ) = zip(*batch)

        lengths = torch.tensor(lengths).flatten()

        swfeatures = pad_sequence(swfeatures, batch_first=True)
        swfeatures = swfeatures.view(swfeatures.shape[0], swfeatures.shape[1], -1)

        sfeatures = pad_sequence(sfeatures, batch_first=True)
        sfeatures = sfeatures.view(sfeatures.shape[0], sfeatures.shape[1], -1)

        features = torch.cat(
            [
                swfeatures,
                sfeatures,
            ],
            dim=2,
        )

        return lengths, features

    # Определим длину каждой последовательности и необходимое количество батчей
    sequence_length = 25
    num_sequences = (
        landmarks_tensor.shape[0] + sequence_length - 1
    ) // sequence_length  # Округление вверх

    # Разделим данные на последовательности по 25 элементов
    sequences = []

    for i in range(num_sequences):
        start_idx = i * sequence_length
        end_idx = min(start_idx + sequence_length, landmarks_tensor.shape[0])

        # Получаем последовательность и вычисляем её истинную длину
        seq_landmarks = landmarks_tensor[start_idx:end_idx]
        seq_world_landmarks = world_landmarks_tensor[start_idx:end_idx]
        true_length = seq_landmarks.shape[0]

        # Дополняем последовательности нулями до длины 25, если они короче
        if true_length < sequence_length:
            padding = torch.zeros(sequence_length - true_length, 33, 3)
            seq_landmarks = torch.cat([seq_landmarks, padding], dim=0)
            seq_world_landmarks = torch.cat([seq_world_landmarks, padding], dim=0)

        # Добавляем последовательности и их длины в список
        sequences.append((true_length, seq_world_landmarks, seq_landmarks))

    lengths_tensor, features = collate_ml(sequences)

    print(f"{features.shape=}")
    print(f"{lengths_tensor.shape=}")
    print(f"{lengths_tensor=}")

    print(f"{features[0]=}")

    lengths_batch, swfeatures_batch = lengths_tensor.clone(), features.clone()
    print(f"{lengths_batch.shape=}")
    print(f"{swfeatures_batch.shape=}")
    print(f"{swfeatures_batch[0]=}")

    max_len = lengths_batch.max()
    mask = torch.arange(max_len).expand(
        len(lengths_batch), max_len
    ) < lengths_batch.unsqueeze(1)

    predicted_labels, predicted_probs = INFERENCE_ELEMENTS.predict(
        features=swfeatures_batch,
        lengths=lengths_batch,
        mask=mask,
    )

    return predicted_labels, predicted_probs
