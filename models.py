import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedLengthCRNN(nn.Module):
    def __init__(self, num_classes, seq_length, input_channels=1):
        super(FixedLengthCRNN, self).__init__()
        self.seq_length = seq_length
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d((2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3))

        # The rnn_input_size and the calculation of H and W depend on the size of the input images
        # and the architecture of the network. It needs to be calculated based on the output size
        # after the convolutional and pooling layers.
        self.rnn_input_size = 64  # Placeholder value, needs to be calculated properly
        self.rnn_hidden_size = 128

        self.lstm = nn.LSTM(
            self.rnn_input_size,
            self.rnn_hidden_size,
            bidirectional=True,
            batch_first=True,
        )

        # Adjust the linear layer to accommodate variable sequence length and number of classes
        self.fc = nn.Linear(
            self.rnn_hidden_size * 2, self.seq_length * self.num_classes
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))

        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2)  # Change to (batch, width, channels, height)
        x = x.view(
            batch_size, width, -1
        )  # Flatten the channels and height into a single dimension

        x, _ = self.lstm(x)
        x = self.fc(x)

        # Reshape to (batch_size, seq_length, num_classes) for sequence predictions
        x = x.view(batch_size, self.seq_length, self.num_classes)

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
