from trainer import Trainer
import torch

trainer = Trainer(
    env_name="ALE/Pong-ram-v5", render=True, episodes=100000, batch_size=64, gamma=0.99, epsilon_start=0.1, epsilon_end=0.001, exploration_percentage=1, learning_rate=3e-4, 
    fc_num=3, fc_neuron_nums=[32, 64, 32], tau=0.005)
trainer.agent.policy_net.load_state_dict(torch.load(f"dqn/{'Pong-ram-v5'}_7000_.ckpt"))
trainer.agent.evaluate(trainer.env_name, 10)