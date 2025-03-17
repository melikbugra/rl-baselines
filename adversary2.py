import gymnasium as gym
from rl_baselines.policy_based.sac.sac import SAC
import continuous_maze_env
import torch
from rl_baselines.utils.base_classes.base_experience_replay import Transition
import numpy as np
from rl_baselines.common.env_wrappers import make_atari_env, make_box2d_viz_env
import tqdm
from rl_baselines.utils import MLFlowLogger


def state_to_torch(state: np.ndarray, device: str) -> torch.Tensor:
    return (
        torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).view(1, -1)
    )


def train():

    mlflow_logger = MLFlowLogger(
        mlflow_tracking_uri="https://mlflow.melikbugraozcelik.com/"
    )

    MAX_STEPS = 1000
    Ka = 10
    Kp = 10
    N = 1000
    Ha = MAX_STEPS // 4
    Hp = (MAX_STEPS // 4) * 3

    # env = make_box2d_viz_env(
    #     "ContinuousMazeViz-v0",
    #     level="level_one",
    #     max_steps=MAX_STEPS,
    #     random_start=True,
    # )

    env = gym.make(
        "ContinuousMaze-v0",
        level="level_three",
        max_steps=MAX_STEPS,
        random_start=False,
    )
    render = False

    adversary = SAC(
        env=env,
        learning_rate=3e-4,
        network_type="mlp",
        device="cpu",
        learning_starts=100000,
        # experience_replay_size=10000,
        network_arch=[128, 128],
    )

    protagonist = SAC(
        env=env,
        learning_rate=3e-4,
        network_type="mlp",
        device="cpu",
        learning_starts=100000,
        # experience_replay_size=10000,
        network_arch=[128, 128],
    )

    mlflow_logger.define_experiment_and_run(
        params_to_log={
            "max_steps": MAX_STEPS,
            "N": N,
            "Ka": Ka,
            "Kp": Kp,
            "Ha": Ha,
            "Hp": Hp,
            "device": "cpu",
            "learning_rate": 3e-4,
            "network_type": "mlp",
            "network_arch": [128, 128],
        },
        env=env,
        algo_name="Adversarial SAC",
    )

    for i in tqdm.tqdm(
        range(N),
        desc="Training Progress",
        position=0,
        bar_format="{desc}: {percentage:1.0f}%|{bar:50}| {n_fmt}/{total_fmt} [ETA: {remaining}, {rate_fmt}]",
        colour="green",
    ):
        total_steps = 0
        if i % 10 == 0 and i != 0:
            render = True
        else:
            render = False

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
                total_steps += 1
                adv_terminates = False
                action = adversary.agent.select_action(state)
                if adversary.agent.action_type == "discrete":
                    action_to_env = action.item()
                else:
                    action_to_env = action.cpu().numpy().flatten().tolist()
                observation, _, terminated, truncated, _ = env.step(action_to_env)
                if render:
                    env.render()
                # reward = get_adversary_reward(observation, protagonist)
                reward = 0
                adv_score += reward
                # tqdm.tqdm.write(f"Adversary reward: {reward}")
                reward = torch.tensor([reward], device=adversary.device)
                if terminated:
                    next_state = None
                else:
                    next_state = state_to_torch(observation, device=adversary.device)
                done = terminated or truncated
                if j == Ha - 1:
                    last_adv_state = state
                    last_adv_action = action
                    last_adv_next_state = next_state
                    last_adv_done = done
                else:
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
                    adv_terminates = True
                    reward = -1
                    adv_score += reward
                    reward = torch.tensor([reward], device=adversary.device)
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

                    # print(f"Random adversary terminated episode at step {i}, restarting...")
                if not done:
                    state = next_state
                else:
                    break

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
                total_steps += 1
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
                if not done:
                    state = next_state
                else:
                    if not adv_terminates:
                        last_adv_reward = -prt_score
                        adv_score += last_adv_reward
                        adv_scores.append(adv_score)
                        last_adv_reward = torch.tensor(
                            [last_adv_reward], device=adversary.device
                        )

                        last_adv_transition = Transition(
                            state=last_adv_state,
                            action=last_adv_action,
                            next_state=last_adv_next_state,
                            reward=last_adv_reward,
                            done=last_adv_done,
                        )

                        adversary.agent.experience_replay.push(
                            transition=last_adv_transition,
                        )
                    break
            prt_scores.append(prt_score)

            mlflow_logger.log_metric(
                key="adv_train_prt_score",
                value=prt_score,
                step=total_steps,
            )
            mlflow_logger.log_metric(
                key="adv_train_adv_score",
                value=adv_score,
                step=total_steps,
            )
            mlflow_logger.log_metric(
                key="adv_train_adv_actor_loss",
                value=adversary.agent.writer.avg_actor_loss,
                step=total_steps,
            )
            mlflow_logger.log_metric(
                key="adv_train_adv_critic_loss",
                value=adversary.agent.writer.avg_critic_loss,
                step=total_steps,
            )
            mlflow_logger.log_metric(
                key="adv_train_prt_actor_loss",
                value=protagonist.agent.writer.avg_actor_loss,
                step=total_steps,
            )
            mlflow_logger.log_metric(
                key="adv_train_prt_critic_loss",
                value=protagonist.agent.writer.avg_critic_loss,
                step=total_steps,
            )

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
            adv_terminates = False
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
                total_steps += 1
                action = adversary.agent.select_greedy_action(state)
                if adversary.agent.action_type == "discrete":
                    action_to_env = action.item()
                else:
                    action_to_env = action.cpu().numpy().flatten().tolist()
                observation, _, terminated, truncated, _ = env.step(action_to_env)
                if render:
                    env.render()
                reward = 0
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
                    adv_terminates = True
                    reward = -1
                    adv_score += reward
                    adv_scores.append(adv_score)
                    break
                    # print(f"Random adversary terminated episode at step {i}, restarting...")
                if not done:
                    state = next_state
                else:
                    break

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
                total_steps += 1
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
                if not done:
                    state = next_state
                else:
                    if not adv_terminates:
                        last_adv_reward = -prt_score
                        adv_score += last_adv_reward
                        adv_scores.append(adv_score)
                    break

            prt_scores.append(prt_score)

            mlflow_logger.log_metric(
                key="prt_train_prt_score",
                value=prt_score,
                step=total_steps,
            )
            mlflow_logger.log_metric(
                key="prt_train_adv_score",
                value=adv_score,
                step=total_steps,
            )
            mlflow_logger.log_metric(
                key="prt_train_adv_actor_loss",
                value=adversary.agent.writer.avg_actor_loss,
                step=total_steps,
            )
            mlflow_logger.log_metric(
                key="prt_train_adv_critic_loss",
                value=adversary.agent.writer.avg_critic_loss,
                step=total_steps,
            )
            mlflow_logger.log_metric(
                key="prt_train_prt_actor_loss",
                value=protagonist.agent.writer.avg_actor_loss,
                step=total_steps,
            )
            mlflow_logger.log_metric(
                key="prt_train_prt_critic_loss",
                value=protagonist.agent.writer.avg_critic_loss,
                step=total_steps,
            )

        tqdm.tqdm.write(
            f"\tMean protagonist score: {sum(prt_scores) / len(prt_scores)}"
        )

        tqdm.tqdm.write(
            f"\tMean adversary score: {sum(adv_scores) / len(adv_scores)}\n"
        )

        if i % 10 == 0 and i != 0:
            adversary.save(save_path=f"models/adversary_sac_{i}")
            protagonist.save(save_path=f"models/protagonist_sac_{i}")
            mlflow_logger.log_artifact(
                local_path=f"models/adversary_sac_{i}.ckpt",
                artifact_path="models",
            )
            mlflow_logger.log_artifact(
                local_path=f"models/protagonist_sac_{i}.ckpt",
                artifact_path="models",
            )


if __name__ == "__main__":
    train()
