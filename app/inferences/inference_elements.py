import random

import numpy as np
import torch
import torch.nn.functional as F
from inferences.models.elements.elements_model import BiLSTMModel


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
