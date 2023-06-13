from ray import tune
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def train_mnist(config):
    # Data loading and transformation
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    # Define the model, loss and optimizer
    model = nn.Sequential(nn.Linear(784, 128),
                          nn.ReLU(),
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64, 10),
                          nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"])

    # Setup Ignite trainer and evaluator
    trainer = create_supervised_trainer(model, optimizer, criterion)
    evaluator = create_supervised_evaluator(model, metrics={"accuracy": Accuracy(), "loss": Loss(criterion)})

    # Run training
    trainer.run(trainloader, max_epochs=5)

    # Run evaluation
    evaluator.run(trainloader)
    metrics = evaluator.state.metrics
    acc = metrics["accuracy"]
    loss = metrics["loss"]

    # Send the metrics to Tune to track the training process
    tune.report(loss=loss, accuracy=acc)


analysis = tune.run(
    train_mnist,
    config={"lr": tune.grid_search([0.001, 0.01, 0.1])}
)

print("Best config: ", analysis.get_best_config(metric="loss", mode="min"))
