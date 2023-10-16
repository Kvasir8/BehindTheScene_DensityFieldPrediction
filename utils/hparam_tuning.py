
## Un-used code, but maybe useful template for tuning

# import itertools
# import random
# from copy import deepcopy

# from ignite.contrib.handlers import ProgressBar
# from ignite.engine import Engine, Events
# from ignite.handlers import Checkpoint, DiskSaver
# from ignite.metrics import Loss, Metric
# from ignite.utils import convert_tensor

# def random_search(config, param_ranges, num_configs):
#     """
#     Perform random search for hyperparameter tuning.

#     Args:
#         config (dict): Base configuration.
#         param_ranges (dict): Dictionary with parameter names as keys and ranges as values.
#         num_configs (int): Number of configurations to sample.

#     Returns:
#         list: List of configurations to train.
#     """
#     configs = []

#     for _ in range(num_configs):
#         sampled_config = deepcopy(config)

#         for param, value_range in param_ranges.items():
#             sampled_config[param] = random.choice(value_range)

#         configs.append(sampled_config)

#     return configs

# def grid_search(config, param_grid):
#     """
#     Perform grid search for hyperparameter tuning.

#     Args:
#         config (dict): Base configuration.
#         param_grid (dict): Dictionary with parameter names as keys and grid values as values.

#     Returns:
#         list: List of configurations to train.
#     """
#     keys, values = zip(*param_grid.items())
#     configs = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

#     for c in configs:
#         for key, value in c.items():
#             config[key] = value

#     return configs

# def create_trainer(model, optimizer, criterion, device):
#     """
#     Create an Ignite trainer engine.

#     Args:
#         model (nn.Module): The model to train.
#         optimizer (torch.optim.Optimizer): The optimizer.
#         criterion (callable): The loss function.
#         device (str): Device to perform computations on.

#     Returns:
#         Engine: The trainer engine.
#     """
#     def _update_fn(engine, batch):
#         model.train()
#         optimizer.zero_grad()

#         x, y = convert_tensor(batch, device=device)
#         output = model(x)
#         loss = criterion(output, y)

#         loss.backward()
#         optimizer.step()

#         return loss.item()

#     trainer = Engine(_update_fn)

#     return trainer

# def train(config):
#     """
#     Train the model with a specific configuration.

#     Args:
#         config (dict): Configuration for training.
#     """
#     # Your training code here

# # Load the base configuration
# base_config = {
#     # Your base configuration here
# }

# # Define the parameter ranges for random search
# param_ranges = {
#     # Parameter ranges for random search
# }

# # Define the parameter grid for grid search
# param_grid = {
#     # Parameter grid for grid search
# }

# # Number of configurations for random search
# num_random_configs = 10

# # Perform random search
# random_configs = random_search(base_config, param_ranges, num_random_configs)

# # Perform grid search
# grid_configs = grid_search(base_config, param_grid)

# # Combine the configurations
# all_configs = random_configs + grid_configs

# # Start the training process for each configuration
# for config in all_configs:
#     # Initialize the model, optimizer, criterion, and device
#     model = initialize_model(config)
#     optimizer = initialize_optimizer(config)
#     criterion = initialize_criterion(config)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Create the trainer engine
#     trainer = create_trainer(model, optimizer, criterion, device)

#     # Attach metrics if needed
#     # metrics = initialize_metrics(config)
#     # for metric in metrics:
#     #     metric.attach(trainer, metric_name)

#     # Attach progress bar
#     pbar = ProgressBar(persist=False)
#     pbar.attach(trainer, metric_names='all')

#     # Start the training process
#     trainer.run(train_loader, max_epochs=config['num_epochs'])

#     # Save the trained model
#     torch.save(model.state_dict(), config['output_path'])
