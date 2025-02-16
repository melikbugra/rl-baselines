import gymnasium as gym
from rl_baselines.policy_based.ppo.ppo import PPO
import continuous_maze_env
import torch
from rl_baselines.utils.base_classes.base_experience_replay import Transition
import numpy as np


def get_adversary_reward(state, protagonist: PPO):
    state_tensor = torch.tensor(
        state, dtype=torch.float32, device=protagonist.device
    ).unsqueeze(0)

    sampled_q_values = []
    with torch.no_grad():
        for _ in range(10):

            _, state_value = protagonist.agent.net(state_tensor)

            sampled_q_values.append(state_value[0])

    # Stack and average the Q-values to get the approximate V(s)
    v_est = torch.mean(torch.stack(sampled_q_values), dim=0)
    return v_est.cpu().item()


def state_to_torch(state: np.ndarray, device: str) -> torch.Tensor:
    return (
        torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).view(1, -1)
    )


def train():
    MAX_STEPS = 1000
    Ka = 5
    Kp = 5
    N = 100
    Ha = MAX_STEPS // 2
    Hp = MAX_STEPS // 2

    env = gym.make(
        "ContinuousMaze-v0", level="level_two", max_steps=MAX_STEPS, random_start=True
    )

    adversary = PPO(
        env=env,
        learning_rate=3e-4,
        network_arch=[128, 128],
        network_type="actor_mlp_critic_mlp",
        device="cpu",
        n_epochs=1,
        gradient_clipping_value=10,
    )

    protagonist = PPO(
        env=env,
        learning_rate=3e-4,
        network_arch=[128, 128],
        network_type="actor_mlp_critic_mlp",
        device="cpu",
        n_epochs=1,
        gradient_clipping_value=10,
    )

    for i in range(N):
        print(f"Training iteration {i}")
        scores = []
        for x in range(Ka):
            state, _ = env.reset()
            state = state_to_torch(state, device=adversary.device)

            # print("Adversary plays...")
            for j in range(Ha):
                if i % 10 == 0 and x == 0:
                    # env.render()
                    pass
                action = adversary.agent.select_action(state)
                observation, _, terminated, truncated, _ = env.step(action.cpu())
                env.render()
                reward = get_adversary_reward(observation, protagonist)
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
                if terminated:
                    # i = 0
                    state, _ = env.reset()
                    state = state_to_torch(state, device=adversary.device)
                    # print(f"Random adversary terminated episode at step {i}, restarting...")
            adversary.agent.optimize_model(time_step=i)

            # print("Protagonist plays...")
            score = 0
            for k in range(Hp):
                if i % 10 == 0 and k == 0:
                    # env.render()
                    pass
                action = protagonist.agent.select_action(state)
                observation, reward, terminated, truncated, _ = env.step(action.cpu())
                env.render()
                reward = torch.tensor([reward], device=protagonist.device)
                score += reward
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
                if done:
                    break
            scores.append(score)

        # print(f"Training iteration {i}, Kp loop")
        for y in range(Kp):
            state, _ = env.reset()
            state = state_to_torch(state, device=protagonist.device)
            # # # print("Adversary plays...")
            for l in range(Ha):
                if i % 10 == 0 and y == 0:
                    # env.render()
                    pass
                action = adversary.agent.select_action(state)
                observation, _, terminated, truncated, _ = env.step(action.cpu())
                env.render()
                reward = get_adversary_reward(observation, protagonist)
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

                adversary.agent.experience_replay.push(
                    transition=transition,
                )

                if terminated:
                    # i = 0
                    state, _ = env.reset()
                    state = state_to_torch(state, device=protagonist.device)
                    # print(f"Random adversary terminated episode at step {i}, restarting...")

            # # # print("Protagonist plays...")
            score = 0
            for m in range(Hp):
                if i % 10 == 0 and m == 0:
                    # env.render()
                    pass
                action = protagonist.agent.select_action(state)
                observation, reward, terminated, truncated, _ = env.step(action.cpu())
                env.render()
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
                if done:
                    break

            protagonist.agent.optimize_model(time_step=i)
            scores.append(score)
        print(f"\n\tMean score: {sum(scores) / len(scores)}")

    adversary.save("trainings/sac_adversary/adversary")
    protagonist.save("trainings/sac_adversary/protagonist")


if __name__ == "__main__":
    train()
