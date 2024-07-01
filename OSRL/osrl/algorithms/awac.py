from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange
import wandb

from osrl.common.net import (EnsembleDoubleQCritic,
                             SquashedGaussianMLPActorforAWAC)
from fsrl.utils import DummyLogger, WandbLogger
from torch.distributions.beta import Beta

class AWAC(nn.Module):

    """
    Advantage Weighted Actor Critic 
    
    Args:
        Args:
        state_dim (int): dimension of the state space.
        action_dim (int): dimension of the action space.
        max_action (float): Maximum action value.
        a_hidden_sizes (list): List of integers specifying the sizes of the layers in the actor network.
        c_hidden_sizes (list): List of integers specifying the sizes of the layers in the critic network.
        vae_hidden_sizes (int): Number of hidden units in the VAE. sample_action_num (int): Number of action samples to draw. 
        gamma (float): Discount factor for the reward.
        tau (float): Soft update coefficient for the target networks. 
        beta (float): Weight of the KL divergence term.            
        lmbda (float): Weight of the Lagrangian term.
        mmd_sigma (float): Width parameter for the Gaussian kernel used in the MMD loss.
        target_mmd_thresh (float): Target threshold for the MMD loss.
        num_samples_mmd_match (int): Number of samples to use in the MMD loss calculation.
        PID (list): List of three floats containing the coefficients of the PID controller.
        kernel (str): Kernel function to use in the MMD loss calculation.
        num_q (int): Number of Q networks in the ensemble.
        num_qc (int): Number of cost Q networks in the ensemble.
        cost_limit (int): Upper limit on the cost per episode.
        episode_len (int): Maximum length of an episode.
        start_update_policy_step (int): Number of steps to wait before updating the policy.
        device (str): Device to run the model on (e.g. 'cpu' or 'cuda:0'). 
    """

    def __init__(
                self,
                state_dim: int,
                action_dim: int,
                #hidden_dim: int = 256,
                hidden_sizes: list = [256, 256],
                max_action: float = 1.0,
                num_q: int = 1,
                # sample_action_num: int = 10,
                gamma: float = 0.99,
                tau: float = 5e-3,
                awac_lambda: float = 1.0,
                exp_adv_max: float = 100.0,
                episode_len: int = 300,
                #start_update_policy_step: int = 20_000,
                device: str = "cuda:0"
    ): 
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.max_action = max_action

        #self.sample_action_num = sample_action_num
        # self.start_update_policy_step = start_update_policy_step
        self.episode_len = episode_len
        self.device = device
        #self.n_train_steps = 0
        self.num_q = num_q
        self.gamma = gamma
        self.tau = tau
        self.awac_lambda = awac_lambda
        self.exp_adv_max = exp_adv_max
        self.actor = SquashedGaussianMLPActorforAWAC(self.state_dim, self.action_dim,
                                              self.hidden_sizes,
                                              nn.ReLU).to(self.device)
        
        self.critic = EnsembleDoubleQCritic(self.state_dim,
                                            self.action_dim,
                                            self.hidden_sizes,
                                            nn.ReLU,
                                            num_q=self.num_q).to(self.device)
        self.target_critic = deepcopy(self.critic)
    ################ create actor critic model ###############
    def setup_optimizers(self, actor_lr, critic_lr):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def actor_loss(self, states, actions):
        with torch.no_grad():
            #获取演员网络输出的动作

            pi_action, action_log_prob = self.actor(states)
            
            #获取评论家网络输出的状态价值
            v1_list, v2_list = self.critic(states, pi_action)

            v = torch.min(
                torch.stack(v1_list + v2_list, dim = 0),dim = 0
            ).values

            #获取动作价值对
            q1_list, q2_list = self.critic(states, actions)
            q = torch.min(
                torch.stack(q1_list + q2_list, dim = 0),dim = 0
            ).values
            adv = q - v
            weights = torch.clamp_max(
                torch.exp(adv / self.awac_lambda), self.exp_adv_max
            )

        pi_action, action_log_prob = self.actor(states)
        loss_actor = (-action_log_prob * weights).mean()
        # print("action_log_prob",action_log_prob,"weight",weights)
        # print("loss_action",loss_actor,"weight",weights)

        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()

        stats_actor = {
            "loss/actor_loss": loss_actor.item(),
        }

        return loss_actor, stats_actor

    def critic_loss(self, states, next_states, actions, rewards, dones):
        with torch.no_grad():
            next_actions, _ = self.actor(next_states)
            target_q1, target_q2, _, _ = self.target_critic.predict(next_states, next_actions)
            
            q_next = torch.min(target_q1, target_q2)
                
            q_target = rewards + self.gamma * (1.0 - dones) * q_next

        _, _, q1_list, q2_list = self.critic.predict(states, actions)
        # print("q_list:",type(q1_list[0]),type(q2_list))
        q1_loss = self.critic.loss(q_target,q1_list)
        q2_loss = self.critic.loss(q_target,q2_list)

        loss_critic = q1_loss + q2_loss

        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()

        stats_critic = {"loss/critic_loss": loss_critic.item()}

        return loss_critic, stats_critic
    
    def soft_update(self,target: nn.Module, source: nn.Module, tau: float):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)
    
    def target_update(self):
        self.soft_update(self.target_critic, self.critic, self.tau)

    def act(self,
            obs: np.ndarray,
            deterministic: bool = False,
            with_logprob: bool = False):
        """
        Given a single obs, return the action, logp.
        """
        # print("Observation type:", type(obs), "Observation value:", obs)
        if isinstance(obs, tuple):
            obs = obs[0]  # 假设数组是元组中的第一个元素
        obs_tensor = torch.tensor(obs[None, ...], dtype=torch.float32).to(self.device)
        # obs = torch.tensor(obs[None, ...], dtype=torch.float32).to(self.device)
        a, logp_a = self.actor(obs_tensor, deterministic, with_logprob)
        a = a.data.numpy() if self.device == "cpu" else a.data.cpu().numpy()
        logp_a = logp_a.data.numpy() if self.device == "cpu" else logp_a.data.cpu(
        ).numpy()
        return np.squeeze(a, axis=0), np.squeeze(logp_a)
    
class AWACTrainer:
    """
    AWAC Trainer
    
    Args:
        model (BEARL): The BEARL model to be trained.
        env (gym.Env): The OpenAI Gym environment to train the model in.
        logger (WandbLogger or DummyLogger): The logger to use for tracking training progress.
        actor_lr (float): learning rate for actor
        critic_lr (float): learning rate for critic
        alpha_lr (float): learning rate for alpha
        vae_lr (float): learning rate for vae
        reward_scale (float): The scaling factor for the reward signal.
        cost_scale (float): The scaling factor for the constraint cost.
        device (str): The device to use for training (e.g. "cpu" or "cuda").
    """
    def __init__(self,
                 model: AWAC,
                 env: gym.Env,
                 logger: WandbLogger = DummyLogger(),
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 reward_scale: float = 1.0,
                 tau: float = 5e-3,
                 cost_scale: float = 1.0,
                 device="cuda"):
        
        self.model = model
        self.logger = logger
        self.env = env
        self.reward_scale = reward_scale
        self.cost_scale = cost_scale
        self.tau = tau
        self.device = device
        self.model.setup_optimizers(actor_lr, critic_lr)

    def train_one_step(self, observations, next_observations, actions, rewards, costs,
                   done):
        """
        Trains the model by updating the critic and actor.
        """
        # update actor
        loss_actor, stats_actor = self.model.actor_loss(observations, actions)
        
        # update critic
        loss_critic, stats_critic = self.model.critic_loss(observations,
                                                           next_observations, actions,
                                                           rewards, done)

        self.model.target_update()

        self.logger.store(**stats_critic)
        self.logger.store(**stats_actor)

    def evaluate(self, eval_episodes):
        """
        Evaluates the performance of the model on a number of episodes.
        """
        self.model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []
        for _ in trange(eval_episodes, desc="Evaluating...", leave=False):
            epi_ret, epi_len, epi_cost = self.rollout()
            episode_rets.append(epi_ret)
            episode_lens.append(epi_len)
            episode_costs.append(epi_cost)
        self.model.train()
        return np.mean(episode_rets) / self.reward_scale, np.mean(
            episode_costs) / self.cost_scale, np.mean(episode_lens)
        
    @torch.no_grad()
    def rollout(self):
        """
        Evaluates the performance of the model on a single episode.
        """
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        obs, info = self.env.reset()
        # self.env.render()
        
        for _ in range(self.model.episode_len):
            act, _ = self.model.act(obs, True, True)
            # print(f"action type: {type(act)}, value: {act}")
            obs_next, reward, terminated, truncated, info = self.env.step(act)
            cost = info["cost"] * self.cost_scale
            obs = obs_next
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        return episode_ret, episode_len, episode_cost

