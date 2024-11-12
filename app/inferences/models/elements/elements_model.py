import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super(Attention, self).__init__()
        # Для двунаправленной LSTM
        self.attention = nn.Linear(hidden_dim * 2, 1)
        # Dropout после линейного слоя
        self.dropout = nn.Dropout(dropout)

    def forward(self, lstm_out, mask=None):
        # lstm_out: [batch_size, seq_len, hidden_dim * 2]
        attention_scores = self.attention(lstm_out)
        # attention_scores: [batch_size, seq_len, 1]
        # Применяем Dropout
        attention_scores = self.dropout(attention_scores)

        # Если передана маска, применяем её к attention_scores
        if mask is not None:
            # Маскируем паддинг (где mask == 0)
            attention_scores = attention_scores.masked_fill(
                mask.unsqueeze(-1) == 0, float("-inf")
            )

        attention_weights = torch.softmax(attention_scores, dim=1)
        # attention_weights: [batch_size, seq_len, 1]

        # Возвращаем веса внимания
        return attention_weights


class BiLSTMModel(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
        weight_decay=0.0,
        use_attention=True,  # Флаг для отключения механизма внимания
    ):
        super(BiLSTMModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes
        self.use_attention = use_attention

        # Линейный слой с нужной размерностью input_dim
        self.input_embedding = nn.Linear(self.input_dim, self.hidden_dim)

        # Batch Normalization после линейного слоя
        self.batch_norm = nn.BatchNorm1d(self.hidden_dim)

        # Dropout перед LSTM слоем
        self.input_dropout = nn.Dropout(dropout)

        # Двунаправленная LSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Включаем слой внимания, если включено в параметрах
        if self.use_attention:
            self.attention = Attention(hidden_dim, dropout=dropout)

        # Dropout перед полносвязным слоем
        self.fc_dropout = nn.Dropout(dropout)

        # Линейный слой для преобразования скрытых состояний в классы
        self.fc = nn.Linear(
            hidden_dim * 2, num_classes
        )  # умножаем на 2, так как LSTM двунаправленная

    def forward(
        self,
        features=None,
        # Длина последовательностей кадров
        lengths=None,
        # Добавляем скрытое состояние для передачи между батчами
        hidden_state=None,
        # Добавляем максу паддинга последовательности
        mask=None,
    ):
        # features.shape = [batch_size, seq_len, input_dim] -> input_dim = 512 + 33*3 + 33*3 + .... + 2
        # Преобразуем входные данные через линейный слой
        embedded = self.input_embedding(features.float())

        # Применяем BatchNorm и Dropout после линейного слоя
        embedded = self.batch_norm(embedded.transpose(1, 2)).transpose(1, 2)
        embedded = self.input_dropout(embedded)

        # Упаковываем последовательности для LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Используем скрытое состояние (hidden_state) для инициализации LSTM
        # Проходим через LSTM
        packed_output, hidden_state = self.lstm(packed_input, hidden_state)

        # Распаковываем последовательности обратно
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        if self.use_attention:
            # Применяем внимание
            attention_weights = self.attention(
                lstm_out, mask=mask
            )  # [batch_size, seq_len, 1]
            # Умножаем выходы LSTM на веса внимания
            lstm_out = (
                lstm_out * attention_weights
            )  # [batch_size, seq_len, hidden_dim * 2]

        # Применяем Dropout перед финальным слоем
        lstm_out = self.fc_dropout(lstm_out)

        # Применяем маску для удаления выходов, соответствующих паддингу
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand(
                lstm_out.size()
            )  # [batch_size, max_len, hidden_dim*2]
            lstm_out = lstm_out * mask_expanded  # Убираем паддинг

        # Применяем полносвязный слой к каждому выходу
        out = self.fc(lstm_out)  # [batch_size, seq_len, num_classes]

        return out, hidden_state
