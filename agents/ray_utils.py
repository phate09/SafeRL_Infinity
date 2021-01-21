import torch
from ray.rllib.agents.dqn import DQNTorchPolicy
from ray.rllib.agents.ppo import PPOTorchPolicy


def convert_ray_policy_to_sequential(policy: PPOTorchPolicy) -> torch.nn.Sequential:
    layers_list = []
    for seq_layer in policy.model.torch_sub_model._hidden_layers:
        for layer in seq_layer._modules['_model']:
            print(layer)
            layers_list.append(layer)
    for layer in policy.model.torch_sub_model._modules['_logits']._modules['_model']:
        print(layer)
        layers_list.append(layer)
    sequential_nn = torch.nn.Sequential(*layers_list)
    return sequential_nn
def convert_ray_simple_policy_to_sequential(policy: PPOTorchPolicy) -> torch.nn.Sequential:
    layers_list = []
    for seq_layer in policy.model._hidden_layers:
        for layer in seq_layer._modules['_model']:
            print(layer)
            layers_list.append(layer)
    for layer in policy.model._modules['_logits']._modules['_model']:
        print(layer)
        layers_list.append(layer)
    sequential_nn = torch.nn.Sequential(*layers_list)
    return sequential_nn
def convert_DQN_ray_policy_to_sequential(policy: DQNTorchPolicy) -> torch.nn.Sequential:
    layers_list = []
    for seq_layer in policy.model._modules['torch_sub_model']._modules['_hidden_layers']:
        for layer in seq_layer._modules['_model']:
            print(layer)
            layers_list.append(layer)
    for seq_layer in policy.model._modules['advantage_module']:
        for layer in seq_layer._modules['_model']:
            print(layer)
            layers_list.append(layer)
    sequential_nn = torch.nn.Sequential(*layers_list)
    return sequential_nn

def load_sequential_from_ray(filename: str,trainer):
    trainer.restore(filename)
    return convert_ray_policy_to_sequential(trainer.get_policy())
