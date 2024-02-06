from trainer import Trainer
import torch


def evaluate():
        trainer = Trainer(
                env_name="CartPole-v0", render=False, episodes=100, batch_size=5,
                alpha=3e-4, gamma=0.99, policy_clip=0.2, fc_neuron_nums=[128,128], device="cpu", n_epochs=4, gae_lambda=0.95)
        trainer.agent.actor.load_state_dict(torch.load(f"trained_models/CartPole-v0_latest_actor.ckpt"))
        trainer.agent.critic.load_state_dict(torch.load(f"trained_models/CartPole-v0_latest_critic.ckpt"))

        return trainer.agent.evaluate(trainer.env_name, 100, save_data=True)

if __name__ == "__main__":
        evaluate()