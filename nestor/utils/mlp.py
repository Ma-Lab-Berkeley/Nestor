from typing import Any, Callable, cast

from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F


class MLP(nn.Module):
    """
    Implements a multi-layer perceptron module.

    Attributes:
        input_dim (int): The dimension of the input features.
        hidden_dims (list[int]): The dimensions of the hidden features at each layer.
        output_dim (int): The dimension of the output features.
        activation (Callable[[Tensor], Tensor]): The activation function to use.

    Trainable parameters:
        layers (nn.ModuleList): The layers of the MLP.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "gelu",
    ):
        super().__init__()
        self.input_dim: int = input_dim
        self.hidden_dims: list[int] = hidden_dims
        self.output_dim: int = output_dim

        if len(hidden_dims) == 0:
            layers = [nn.Linear(input_dim, output_dim)]
        else:
            layers = [nn.Linear(input_dim, hidden_dims[0])]
            for i in range(1, len(hidden_dims)):
                layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.layers: list[nn.Linear] = cast(
            list[nn.Linear], nn.ModuleList(layers)
        )

        self.activation: str = activation
        activation_fn = None
        match activation:
            case "relu":
                activation_fn = F.relu
            case "gelu":
                activation_fn = F.gelu
            case "silu":
                activation_fn = F.silu
            case "tanh":
                activation_fn = F.tanh
            case _:
                raise ValueError(f"Invalid activation function: {activation}")

        self._activation_fn: Callable[[Tensor], Tensor] = activation_fn

    def forward(
        self,
        x: Float[Tensor, "... input_dim"],
    ) -> Float[Tensor, "... output_dim"]:
        """
        Computes the output of the MLP.

        Args:
            x (Float[Tensor, "... input_dim"]): The input features.

        Returns:
            Float[Tensor, "... output_dim"]: The output features.
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self._activation_fn(x)
        x = self.layers[-1](x)
        return x

    def to_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the module.

        Returns:
            dict[str, Any]: The configuration of the module. Has keys:
                - input_dim (int): The dimension of the input features.
                - hidden_dims (list[int]): The dimensions of the hidden features at each layer.
                - output_dim (int): The dimension of the output features.
                - activation (str): The activation function to use.
        """
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "output_dim": self.output_dim,
            "activation": self.activation,
        }

    @staticmethod
    def from_config(config: dict[str, Any]) -> "MLP":
        """
        Returns the module from the configuration.

        Args:
            config (dict[str, Any]): The configuration of the module. Has keys:
                - input_dim (int): The dimension of the input features.
                - hidden_dims (list[int]): The dimensions of the hidden features at each layer.
                - output_dim (int): The dimension of the output features.
                - activation (str): The activation function to use.

        Returns:
            MLP: The initialized module from the configuration.
        """
        return MLP(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"],
            output_dim=config["output_dim"],
            activation=config["activation"],
        )
