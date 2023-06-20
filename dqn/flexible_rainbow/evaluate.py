from trainer import Trainer
import torch

trainer = Trainer(
    env_name="SimpleMaze-v1", render=True, episodes=100000, batch_size=64, gamma=0.99, epsilon_start=0.1, epsilon_end=0.001, exploration_percentage=1, learning_rate=3e-4, 
    fc_num=2, fc_neuron_nums=[512,512], tau=0.005)
trainer.agent.policy_net.load_state_dict(torch.load(f"dqn/SimpleMaze-v1_latest.ckpt"))
trainer.agent.evaluate(trainer.env_name, 10)