import torch
import torch.nn as nn


class PointwiseFeedForward(nn.Module):
    """
    Pointwise Feed-Forward Network (FFN) for the SASRec recommender.

    Consists of two linear projection layers mapping hidden_units -> hidden_units,
    with a ReLU activation and dropout. This matches the SASRec paper, which
    maintains the same hidden dimensionality in the intermediate layer.

    Note: Residual connections are not handled within this class and should
    be applied externally by the enclosing block.
    """
    def __init__(self, hidden_units: int, dropout_rate: float):
        super().__init__()
        self.linear1 = nn.Linear(hidden_units, hidden_units)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FFN.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_units)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_units)
        """
        out = self.linear1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        return out
