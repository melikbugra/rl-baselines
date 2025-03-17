import gymnasium as gym
from rl_baselines.policy_based.sac.sac import SAC
import continuous_maze_env
import torch
import numpy as np
import tqdm
from time import sleep


def state_to_torch(state: np.ndarray, device: str) -> torch.Tensor:
    return (
        torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).view(1, -1)
    )


def eval():
    model_checkpoint = 10
    EVAL_EPISODES = 10
    MAX_STEPS = 2500
    Ha = MAX_STEPS // 4
    Hp = (MAX_STEPS // 4) * 3
    env = gym.make(
        "ContinuousMaze-v0",
        level="level_three",
        max_steps=MAX_STEPS,
        random_start=False,
    )

    adversary = SAC(
        env=env,
        experience_replay_size=512,
        device="cpu",
    )
    adversary.load(model_path=f"models/adversary_sac_{model_checkpoint}")

    protagonist = SAC(
        env=env,
        experience_replay_size=512,
        device="cpu",
    )
    protagonist.load(model_path=f"models/protagonist_sac_{model_checkpoint}")

    for i in tqdm.tqdm(
        range(EVAL_EPISODES),
        desc="Evaluation Progress",
        position=0,
        bar_format="{desc}: {percentage:1.0f}%|{bar:50}| {n_fmt}/{total_fmt} [ETA: {remaining}, {rate_fmt}]",
        colour="green",
    ):
        adv_play_pbar = tqdm.tqdm(
            range(Ha),
            desc=f"Adversary Playing Episode {i}",
            leave=False,
            position=1,
            bar_format="{desc}: {percentage:1.0f}%|{bar:50}| {n_fmt}/{total_fmt} ",
            colour="red",
        )
        state, _ = env.reset()
        state = state_to_torch(state, device=adversary.device)

        adv_score = 0
        for j in adv_play_pbar:
            sleep(0.01)
            adv_terminates = False
            action = adversary.agent.select_greedy_action(state)
            if adversary.agent.action_type == "discrete":
                action_to_env = action.item()
            else:
                action_to_env = action.cpu().numpy().flatten().tolist()
            state, _, terminated, truncated, _ = env.step(action_to_env)
            state = state_to_torch(state, device=adversary.device)
            env.render()
            reward = 0

            if terminated:
                reward = -1
                adv_terminates = True
                break

        prt_score = 0
        protagonist_play_pbar = tqdm.tqdm(
            range(Hp),
            desc=f"Protagonist Playing Episode {i}",
            leave=False,
            position=1,
            bar_format="{desc}: {percentage:1.0f}%|{bar:50}| {n_fmt}/{total_fmt} ",
            colour="blue",
        )
        for k in protagonist_play_pbar:
            sleep(0.01)
            action = protagonist.agent.select_greedy_action(state)
            if protagonist.agent.action_type == "discrete":
                action_to_env = action.item()
            else:
                action_to_env = action.cpu().numpy().flatten().tolist()

            state, reward, terminated, truncated, _ = env.step(action_to_env)
            state = state_to_torch(state, device=adversary.device)
            env.render()
            prt_score += reward

            done = terminated or truncated
            if done:
                if not adv_terminates:
                    last_adv_reward = -prt_score
                    adv_score += last_adv_reward
                break

        tqdm.tqdm.write(f"Adversary Score: {adv_score}")
        tqdm.tqdm.write(f"Protagonist Score: {prt_score}")


if __name__ == "__main__":
    eval()
