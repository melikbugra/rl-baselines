from trainer import Trainer
import torch
import json

with open("/home/melikozcelik/rl-projects/rainbow_gym/rainbow/config-speed.json", "rb") as jsn:
    config = json.load(jsn)
trainer = Trainer(
    env_name="highway-fast-v0", render=False, episodes=500, batch_size=64, gamma=0.99, epsilon_start=1, epsilon_end=0.001, exploration_percentage=10, learning_rate=3e-4, 
    fc_num=2, fc_neuron_nums=[512,512], tau=0.005, device="cpu")
trainer.agent.policy_net.load_state_dict(torch.load(f"/home/melikozcelik/rl-projects/rainbow_gym/rainbow/dqn/highway-fast-v0_recent.ckpt"))
trainer.agent.evaluate(trainer.env_name, 10000, config)