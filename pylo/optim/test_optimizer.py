import torch
from torch.optim import Optimizer

class TestOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(TestOptimizer, self).__init__(params, defaults)
        self.network = torch.nn.Linear(10, 2)
        

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)

        return loss

# Example usage
if __name__ == "__main__":
    model = torch.nn.Linear(10, 2)
    optimizer = TestOptimizer(model.parameters(), lr=0.01)

    # Dummy input and target
    input = torch.randn(3, 10)
    target = torch.randn(3, 2)

    criterion = torch.nn.MSELoss()

    for _ in range(100):
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")