import torch
class DQN(torch.nn.Module):
    def __init__(
        self, 
        n_observations, 
        n_actions : int,
        activation : callable = torch.nn.ReLU
    ) -> None:
        super().__init__()

        self.activation = activation()

        self.layer1 = torch.nn.Linear(
            in_features=n_observations, 
            out_features=128
        )
        self.layer2 = torch.nn.Linear(
            in_features=128, 
            out_features=128,
        )
        self.layer3 = torch.nn.Linear(
            in_features=128,
            out_features=n_actions
        )

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)

        return x