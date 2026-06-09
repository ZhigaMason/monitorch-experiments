import argparse
import torch
import torch.nn as nn
import gymnasium as gym
import yaml
from optimizer_utils import get_optimizer

from monitorch.inspector import PyTorchInspector
from monitorch.visualizer import RecorderVisualizer
from monitorch import lens


class SACActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(256, act_dim)
        self.log_std_head = nn.Linear(256, act_dim)

    def forward(self, x):
        feat = self.net(x)
        return self.mu_head(feat), self.log_std_head(feat)


class SACCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(), nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(), nn.Linear(256, 1)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


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

    env = gym.make(config["env_id"])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = SACActor(obs_dim, act_dim)
    critic = SACCritic(obs_dim, act_dim)

    # We create independent optimizer wrappers for the Actor and Critic
    actor_opt = get_optimizer(actor, config["optimizer"], config["lr"])
    critic_opt = get_optimizer(critic, config["optimizer"], config["lr"])

    log_file = args.config_file.split("/")[-1].split(".")[0] + f"_{config['seed']}.pkl"
    actor_inspector = PyTorchInspector(
        lenses=[
            lens.LossMetrics(),
            lens.OutputActivation(),
            lens.ParameterGradientActivation(),
        ],
        module=actor,
        visualizer=RecorderVisualizer("actor_" + log_file),
    )
    critic_inspector = PyTorchInspector(
        lenses=[
            lens.LossMetrics(),
            lens.OutputActivation(),
            lens.ParameterGradientActivation(),
        ],
        module=critic,
        visualizer=RecorderVisualizer("critic_" + log_file),
    )

    obs, _ = env.reset(seed=config["seed"])
    for step in range(config["steps"]):
        # Dummy transition for structural demonstration
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)

        # In a real run, sample from replay buffer
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        act_t = torch.FloatTensor(action).unsqueeze(0)
        rew_t = torch.FloatTensor([reward]).unsqueeze(0)

        # Update Critic
        q1, q2 = critic(obs_t, act_t)
        critic_loss = ((q1 - rew_t) ** 2 + (q2 - rew_t) ** 2).mean()
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        # Update Actor
        mu, _ = actor(obs_t)
        q1_pi, _ = critic(obs_t, torch.tanh(mu))
        actor_loss = -q1_pi.mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        obs = next_obs if not (terminated or truncated) else env.reset()[0]

        actor_inspector.push_loss(actor_loss.item(), train=True)
        critic_inspector.push_loss(critic_loss.item(), train=True)

        if step % 500 == 0:
            actor_inspector.tick()
            critic_inspector.tick()
            print(
                f"Step {step} | C_Loss: {critic_loss.item():.4f} | A_Loss: {actor_loss.item():.4f}"
            )


if __name__ == "__main__":
    main()
