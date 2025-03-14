import gymnasium as gym
from rl_baselines.policy_based.sac.sac import SAC
import continuous_maze_env
import torch
from rl_baselines.utils.base_classes.base_experience_replay import Transition
import numpy as np
from rl_baselines.common.env_wrappers import make_atari_env, make_box2d_viz_env
import tqdm


def get_adversary_reward(state, protagonist: SAC):
    state_tensor = torch.tensor(
        state, dtype=torch.float32, device=protagonist.device
    ).unsqueeze(0)

    v_values = []
    with torch.no_grad():
        for _ in range(10):
            outs, _, _, _, _ = protagonist.agent.net(
                state=state_tensor, actor_pass=True
            )
            mean, std = outs[0]

            noise = torch.randn_like(mean)
            z = mean + std * noise
            action = torch.tanh(z) * protagonist.agent.max_action

            # Get Q-values for the sampled action
            _, q1, q2, _, _ = protagonist.agent.net(
                state=state_tensor, action=action, critic_pass=True
            )

            # Use minimum Q-value (as in SAC)
            min_q = torch.min(q1, q2)

            # Calculate log probability of action
            log_prob_gauss = -0.5 * (
                ((z - mean) / std) ** 2 + 2 * torch.log(std) + np.log(2 * np.pi)
            )
            log_prob_gauss = log_prob_gauss.sum(dim=-1, keepdim=True)
            log_prob_policy = log_prob_gauss - (
                1 - torch.tanh(z) ** 2 + 1e-6
            ).log().sum(dim=-1, keepdim=True)

            # V(s) = Q(s,a) - α * log_prob(a|s)
            alpha = torch.exp(protagonist.agent.log_alpha)
            v_value = min_q - alpha * log_prob_policy
            v_values.append(v_value)

    # Average the sampled V-values
    v_est = torch.mean(torch.stack(v_values), dim=0)
    return -v_est.cpu().item()


def state_to_torch(state: np.ndarray, device: str) -> torch.Tensor:
    return (
        torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).view(1, -1)
    )


def train():
    MAX_STEPS = 1000
    Ka = 10
    Kp = 10
    N = 100
    Ha = MAX_STEPS // 4
    Hp = (MAX_STEPS // 4) * 3

    # env = make_box2d_viz_env(
    #     "ContinuousMazeViz-v0",
    #     level="level_one",
    #     max_steps=MAX_STEPS,
    #     random_start=True,
    # )

    env = gym.make(
        "ContinuousMaze-v0", level="level_one", max_steps=MAX_STEPS, random_start=False
    )
    render = False

    adversary = SAC(
        env=env,
        learning_rate=3e-4,
        network_type="mlp",
        device="cpu",
        learning_starts=0,
        experience_replay_size=10000,
        network_arch=[32, 32],
    )

    protagonist = SAC(
        env=env,
        learning_rate=3e-4,
        network_type="mlp",
        device="cpu",
        learning_starts=0,
        experience_replay_size=10000,
        network_arch=[32, 32],
    )

    for i in tqdm.tqdm(
        range(N),
        desc="Training Progress",
        position=0,
        bar_format="{desc}: {percentage:1.0f}%|{bar:50}| {n_fmt}/{total_fmt} [ETA: {remaining}, {rate_fmt}]",
        colour="green",
    ):
        if i == 50:
            render = True

        if adversary.agent.steps_done == 500000:
            tqdm.tqdm.write("\n\nAdversary learning starts...\n\n")

        adv_train_pbar = tqdm.tqdm(
            range(Ka),
            desc=f"Training Adversary (Iter {i})",
            leave=False,
            position=1,
            bar_format="{desc}: {percentage:1.0f}%|{bar:50}| {n_fmt}/{total_fmt} ",
            colour="red",
        )
        prt_scores = []
        adv_scores = []
        tqdm.tqdm.write("Adversary training...")
        for x in adv_train_pbar:
            state, _ = env.reset()
            state = state_to_torch(state, device=adversary.device)

            adv_play_pbar = tqdm.tqdm(
                range(Ha),
                desc=f"Adversary Playing (Iter {i}, Ka {x})",
                leave=False,
                position=2,
                bar_format="{desc}: {percentage:1.0f}%|{bar:50}| {n_fmt}/{total_fmt} ",
                colour="red",
            )
            adv_score = 0
            for j in adv_play_pbar:
                action = adversary.agent.select_action(state)
                if adversary.agent.action_type == "discrete":
                    action_to_env = action.item()
                else:
                    action_to_env = action.cpu().numpy().flatten().tolist()
                observation, _, terminated, truncated, _ = env.step(action_to_env)
                if render:
                    env.render()
                reward = get_adversary_reward(observation, protagonist)
                adv_score += reward
                # tqdm.tqdm.write(f"Adversary reward: {reward}")
                reward = torch.tensor([reward], device=adversary.device)
                if terminated:
                    next_state = None
                else:
                    next_state = state_to_torch(observation, device=adversary.device)
                done = terminated or truncated
                transition = Transition(
                    state=state,
                    action=action,
                    next_state=next_state,
                    reward=reward,
                    done=done,
                )

                adversary.agent.experience_replay.push(
                    transition=transition,
                )
                adversary.agent.optimize_model(time_step=i)
                if terminated:
                    # i = 0
                    state, _ = env.reset()
                    state = state_to_torch(state, device=adversary.device)
                    # print(f"Random adversary terminated episode at step {i}, restarting...")
            adv_scores.append(adv_score)

            # print("Protagonist plays...")
            prt_score = 0

            protagonist_play_pbar = tqdm.tqdm(
                range(Hp),
                desc=f"Protagonist Playing (Iter {i}, Ka {x})",
                leave=False,
                position=2,
                bar_format="{desc}: {percentage:1.0f}%|{bar:50}| {n_fmt}/{total_fmt} ",
                colour="blue",
            )
            for k in protagonist_play_pbar:
                action = protagonist.agent.select_greedy_action(state)
                if protagonist.agent.action_type == "discrete":
                    action_to_env = action.item()
                else:
                    action_to_env = action.cpu().numpy().flatten().tolist()
                observation, reward, terminated, truncated, _ = env.step(action_to_env)
                if render:
                    env.render()

                prt_score += reward
                # reward = torch.tensor([reward], device=protagonist.device)
                if terminated:
                    next_state = None
                else:
                    next_state = state_to_torch(observation, device=adversary.device)
                done = terminated or truncated
                # transition = Transition(
                #     state=state,
                #     action=action,
                #     next_state=next_state,
                #     reward=reward,
                #     done=done,
                # )

                # protagonist.agent.experience_replay.push(
                #     transition=transition,
                # )
                if done:
                    break
            prt_scores.append(prt_score)

        tqdm.tqdm.write(
            f"\tMean protagonist score: {sum(prt_scores) / len(prt_scores)}"
        )

        tqdm.tqdm.write(
            f"\tMean adversary score: {sum(adv_scores) / len(adv_scores)}\n"
        )

        protagonist_train_pbar = tqdm.tqdm(
            range(Kp),
            desc=f"Training Protagonist (Iter {i})",
            leave=False,
            position=1,
            bar_format="{desc}: {percentage:1.0f}%|{bar:50}| {n_fmt}/{total_fmt} ",
            colour="blue",
        )
        tqdm.tqdm.write("Protagonist training...")
        prt_scores = []
        adv_scores = []
        for y in protagonist_train_pbar:
            state, _ = env.reset()
            state = state_to_torch(state, device=protagonist.device)
            adv_play_pbar = tqdm.tqdm(
                range(Ha),
                desc=f"Adversary Playing (Iter {i}, Kp {y})",
                leave=False,
                position=2,
                bar_format="{desc}: {percentage:1.0f}%|{bar:50}| {n_fmt}/{total_fmt} ",
                colour="red",
            )
            adv_score = 0
            for l in adv_play_pbar:
                action = adversary.agent.select_greedy_action(state)
                if adversary.agent.action_type == "discrete":
                    action_to_env = action.item()
                else:
                    action_to_env = action.cpu().numpy().flatten().tolist()
                observation, _, terminated, truncated, _ = env.step(action_to_env)
                if render:
                    env.render()
                reward = get_adversary_reward(observation, protagonist)
                adv_score += reward
                # reward = torch.tensor([reward], device=protagonist.device)
                # if terminated:
                #     next_state = None
                # else:
                #     next_state = state_to_torch(observation, device=adversary.device)

                # done = terminated or truncated

                # transition = Transition(
                #     state=state,
                #     action=action,
                #     next_state=next_state,
                #     reward=reward,
                #     done=done,
                # )

                # adversary.agent.experience_replay.push(
                #     transition=transition,
                # )

                if terminated:
                    # i = 0
                    state, _ = env.reset()
                    state = state_to_torch(state, device=protagonist.device)
                    # print(f"Random adversary terminated episode at step {i}, restarting...")
            adv_scores.append(adv_score)

            # # # print("Protagonist plays...")
            prt_score = 0
            protagonist_play_pbar = tqdm.tqdm(
                range(Hp),
                desc=f"Protagonist Playing (Iter {i}, Kp {y})",
                leave=False,
                position=2,
                bar_format="{desc}: {percentage:1.0f}%|{bar:50}| {n_fmt}/{total_fmt} ",
                colour="blue",
            )
            for m in protagonist_play_pbar:
                action = protagonist.agent.select_action(state)
                if protagonist.agent.action_type == "discrete":
                    action_to_env = action.item()
                else:
                    action_to_env = action.cpu().numpy().flatten().tolist()
                observation, reward, terminated, truncated, _ = env.step(action_to_env)
                if render:
                    env.render()
                prt_score += reward
                reward = torch.tensor([reward], device=protagonist.device)
                if terminated:
                    next_state = None
                else:
                    next_state = state_to_torch(observation, device=adversary.device)
                done = terminated or truncated
                transition = Transition(
                    state=state,
                    action=action,
                    next_state=next_state,
                    reward=reward,
                    done=done,
                )

                protagonist.agent.experience_replay.push(
                    transition=transition,
                )
                protagonist.agent.optimize_model(time_step=i)
                if done:
                    break

            prt_scores.append(prt_score)
        tqdm.tqdm.write(
            f"\tMean protagonist score: {sum(prt_scores) / len(prt_scores)}"
        )

        tqdm.tqdm.write(
            f"\tMean adversary score: {sum(adv_scores) / len(adv_scores)}\n"
        )

    adversary.save(folder="models", checkpoint="last")
    protagonist.save(folder="models", checkpoint="last")


if __name__ == "__main__":
    train()
