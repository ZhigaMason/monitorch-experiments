import argparse
import torch
import torch.nn as nn
import gymnasium as gym
import yaml
import numpy as np
from optimizer_utils import get_optimizer

from monitorch.inspector import PyTorchInspector
from monitorch.visualizer import RecorderVisualizer
from monitorch import lens


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # 2D weight matrices here will be caught by Muon
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        # Naming is critical: 'head' ensures AdamW takes these
        self.actor_head = nn.Linear(64, act_dim)
        self.critic_head = nn.Linear(64, 1)

    def forward(self, x):
        features = self.shared(x)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config-file",
        type=str,
        help="Config file.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    env = gym.make(config["env_id"])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = ActorCritic(obs_dim, act_dim)
    optimizer = get_optimizer(
        model, config["optimizer"], config["lr"], config["weight_decay"]
    )

    log_file = args.config_file.split("/")[-1].split(".")[0] + f"_{config['seed']}.pkl"
    inspector = PyTorchInspector(
        lenses=[
            lens.LossMetrics(),
            lens.OutputActivation(),
            lens.ParameterGradientActivation(),
        ],
        module=model,
        visualizer=RecorderVisualizer(log_file),
    )

    # Minimal PPO interaction loop (simplified for benchmarking optimizers)
    obs, _ = env.reset(seed=config["seed"])
    for step in range(config["total_timesteps"]):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

        logits, value = model(obs_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        next_obs, reward, terminated, truncated, _ = env.step(action.item())

        # PPO Loss components (simplified heavily for demonstration)
        # In a real run, you'd collect trajectories and compute GAE
        log_prob = dist.log_prob(action)
        advantage = reward - value.detach()
        actor_loss = -(log_prob * advantage).mean()
        critic_loss = (reward - value).pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        obs = next_obs if not (terminated or truncated) else env.reset()[0]

        if step % 1000 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | Reward: {reward}")


if __name__ == "__main__":
    main()
