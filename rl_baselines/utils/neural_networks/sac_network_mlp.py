import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from rl_baselines.utils.base_classes.base_neural_network import BaseNeuralNetwork
from rl_baselines.utils.base_classes.base_sac_neural_network import BaseSACNeuralNetwork


class SACNetworkMLP(BaseSACNeuralNetwork):
    def __init__(
        self,
        actor_mlp: BaseNeuralNetwork,
        critic_mlps: list[BaseNeuralNetwork],  # <— YENİ: [head_1, ..., head_H]
        target_critic_mlps: list[BaseNeuralNetwork],  # <— YENİ: hedef başlar
    ):
        super().__init__()
        self.network_type: str = "mlp"
        self.actor = actor_mlp

        # Critics as lists (H heads)
        assert len(critic_mlps) >= 2, "Use at least 2 Q-heads (>= REDQ/Double)"
        assert len(critic_mlps) == len(target_critic_mlps)
        self.critics = nn.ModuleList(critic_mlps)
        self.target_critics = nn.ModuleList(target_critic_mlps)

        self.num_q_heads = len(self.critics)

        self.action_type = actor_mlp.action_type
        self.action_dim = actor_mlp.action_dim  # (act_dim,)

    def forward(
        self,
        state: Tensor,
        action: Tensor = None,
        actor_pass: bool = False,
        critic_pass: bool = False,
        target_pass: bool = False,
    ):
        """
        Dönüş:
          outs: actor çıkışı (veya None)
          q_stack: [B, H] (veya None)  — critics
          q_mean:  [B, 1] (veya None)
          q_std:   [B, 1] (veya None)
          tgt_q_stack, tgt_q_mean, tgt_q_std: target critics (veya None)
        """
        outs = self.actor(state) if actor_pass else None

        def _stack_qs(modules: nn.ModuleList, s: Tensor, a: Tensor):
            # Her head için tek değer [B,1]; sonra [B,H] stokla
            qs = []
            sa = torch.cat([s, a], dim=-1)
            for m in modules:
                q = m(sa)[0]  # senin MLP Base çıktın ilk indexte tensor veriyor.
                qs.append(q)
            q_stack = torch.cat(qs, dim=1)  # [B, H]
            q_mean = q_stack.mean(dim=1, keepdim=True)
            q_std = q_stack.std(dim=1, keepdim=True, unbiased=False)
            return q_stack, q_mean, q_std

        if critic_pass and action is not None:
            q_stack, q_mean, q_std = _stack_qs(self.critics, state, action)
        else:
            q_stack = q_mean = q_std = None

        if target_pass and action is not None:
            tgt_q_stack, tgt_q_mean, tgt_q_std = _stack_qs(
                self.target_critics, state, action
            )
        else:
            tgt_q_stack = tgt_q_mean = tgt_q_std = None

        return outs, q_stack, q_mean, q_std, tgt_q_stack, tgt_q_mean, tgt_q_std
