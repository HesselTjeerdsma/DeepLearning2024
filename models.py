import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedLengthCRNN(nn.Module):
    def __init__(self):
        super(FixedLengthCRNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d((2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3))

        # Assuming the input size is fixed and known, calculate the size here
        # For example, let's assume the output of the last conv layer is (batch_size, 64, H, W)
        # You will need to adjust H and W based on your actual output sizes
        self.rnn_input_size = (
            64  # This needs to be adjusted based on the output shape after conv layers
        )
        self.rnn_hidden_size = 128

        self.lstm = nn.LSTM(
            self.rnn_input_size,
            self.rnn_hidden_size,
            bidirectional=True,
            batch_first=True,
        )

        # For a fixed length of 4 and 36 possible characters, adjust the linear layer
        self.fc = nn.Linear(
            self.rnn_hidden_size * 2, 4 * 36
        )  # 128 * 2 for bidirectional

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))

        # Adjust dimensions before feeding into RNN
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2)  # Change to (batch, width, channels, height)
        x = x.view(
            batch_size, width, -1
        )  # Flatten the channels and height into a single dimension

        x, _ = self.lstm(x)
        x = self.fc(x)

        # Reshape to (batch_size, 4, 36) to get predictions for each of the 4 characters
        x = x.view(batch_size, 4, 36)

        return x


# Model architecture
class VariableLengthCRNN(nn.Module):
    def __init__(self):
        super(VariableLengthCRNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), activation="relu")
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), activation="relu")
        self.pool2 = nn.MaxPool2d((2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), activation="relu")
        self.rnn_input_size = (
            64 * 15 * 40
        )  # Update according to the output shape after conv layers
        self.lstm = nn.LSTM(
            self.rnn_input_size, 128, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(256, 11)  # 128 * 2 for bidirectional, 10 + 1 for classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


# Custom CTC Loss
# In PyTorch, CTC Loss is already implemented and can be used directly.
# You will need to provide logits from your model, target (labels), input_lengths, and target_lengths
ctc_loss = nn.CTCLoss()

# Example on how to calculate CTC Loss
# logits: tensor of shape (T, N, C) where T is the maximum sequence length, N is the batch size, C is the number of classes (including blank)
# labels: tensor of shape (sum(target_lengths))
# input_lengths: tensor of size (N)
# target_lengths: tensor of size (N)
# loss = ctc_loss(logits, labels, input_lengths, target_lengths)

# Note: Make sure to use log_softmax on your output logits before passing them to CTC loss
