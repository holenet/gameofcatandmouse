import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing


from collections import defaultdict

import matplotlib
matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv, ParallelEnv)
from torchrl.envs.libs.gym import GymEnv, GymWrapper
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from env import CatMouseEnv, CatMouseWrapper

import sys
try:
    speed_factor = float(sys.argv[1])
    ckpt_path = sys.argv[2]
    print(speed_factor, ckpt_path)
except:
    speed_factor = 4.0
    ckpt_path = 'ckpt.pth'

# Define Hyperparameters

if __name__ == '__main__':
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    num_cells = 256  # number of cells in each layer i.e. output dim.
    lr = 3e-4
    max_grad_norm = 1.0

    frames_per_batch = 1000
    # For a complete training, bring the number of frames up to 1M
    total_frames = 1_000_000_000
    # total_frames = 50_000

    sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = 10  # optimization steps per batch of data collected
    clip_epsilon = (
        0.2  # clip value for PPO loss: see the equation in the intro for more context.
    )
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-4


    # speed_factor = 4.49


    # Define an environment

    # base_env = GymEnv("InvertedDoublePendulum-v4", device=device)
    # base_env = GymWrapper(CatMouseWrapper(CatMouseEnv(render_mode='human', speed_factor=0.1)))

    make_env = lambda: TransformedEnv(
        GymWrapper(CatMouseWrapper(CatMouseEnv(render_mode='human', speed_factor=speed_factor))),
        Compose(
            # normalize observations
            # ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    env = make_env()
    # env = ParallelEnv(2, make_env)

    # env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

    # print("normalization constant shape:", env.transform[0].loc.shape)

    # print("observation_spec:", env.observation_spec)
    # print("reward_spec:", env.reward_spec)
    # print("input_spec:", env.input_spec)
    # print("action_spec (as defined by input_spec):", env.action_spec)

    # check_env_specs(env)

    # rollout = env.rollout(1000)
    # print("rollout of three steps:", rollout)
    # print("Shape of the rollout TensorDict:", rollout.batch_size)


    # Policy

    actor_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        # nn.Tanh(),
        nn.LeakyReLU(),
        nn.LazyLinear(num_cells, device=device),
        # nn.Tanh(),
        nn.LeakyReLU(),
        nn.LazyLinear(num_cells, device=device),
        # nn.Tanh(),
        nn.LeakyReLU(),
        nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
        NormalParamExtractor(),
    )
    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )
    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env.action_spec.space.low,
            "max": env.action_spec.space.high,
        },
        return_log_prob=True,
        # we'll need the log-prob for the numerator of the importance weights
    )


    # Value network

    value_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        # nn.Tanh(),
        nn.LeakyReLU(),
        nn.LazyLinear(num_cells, device=device),
        # nn.Tanh(),
        nn.LeakyReLU(),
        nn.LazyLinear(num_cells, device=device),
        # nn.Tanh(),
        nn.LeakyReLU(),
        nn.LazyLinear(1, device=device),
    )
    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    print("Running policy:", policy_module(env.reset()))
    print("Running value:", value_module(env.reset()))
    # env.close()


    # Data Collector

    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,
    )
    # if __name__ == '__main__':
    #     from multiprocessing import freeze_support
    #     freeze_support()
    # collector = MultiSyncDataCollector(
    #     create_env_fn=[make_env] * 16,
    #     policy=policy_module,
    #     frames_per_batch=frames_per_batch,
    #     total_frames=total_frames,
    #     split_trajs=False,
    #     device=device,
    # )


    # Replay buffer

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )


    # Loss function

    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        # these keys match by default but we set this for completeness
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )


    # Load model
    try:
        sd_policy, sd_value, sd_optim = torch.load(ckpt_path, map_location=device)
        policy_module.load_state_dict(sd_policy)
        value_module.load_state_dict(sd_value)
        optim.load_state_dict(sd_optim)
        print('Model Loaded')
        try:
            if sys.argv[3] == '-i':
                try:
                    for _ in range(100):
                        env.rollout(1000, policy_module, auto_reset=True)
                except KeyboardInterrupt:
                    exit()
                finally:
                    exit()
        except:
            pass
    except Exception as e:
        print('Model Load Failed', e)
        pass

    # Training loop

    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""

    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for i, tensordict_data in enumerate(collector):
        # we now have a batch of data to work with. Let's learn something from it.
        for _ in range(num_epochs):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())

        # Save model
        torch.save([policy_module.state_dict(), value_module.state_dict(), optim.state_dict()], ckpt_path)

        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        if i % 10 == 0:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our ``env`` horizon).
            # The ``rollout`` method of the ``env`` can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = env.rollout(1000, policy_module)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout
        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(logs["reward"])
    plt.title("training rewards (average)")
    plt.subplot(2, 2, 2)
    plt.plot(logs["step_count"])
    plt.title("Max step count (training)")
    plt.subplot(2, 2, 3)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Return (test)")
    plt.subplot(2, 2, 4)
    plt.plot(logs["eval step_count"])
    plt.title("Max step count (test)")
    plt.show()
