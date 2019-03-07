"""
Implementation of ACER.
"""
import argparse
from collections import deque

import gym
import numpy as np

from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim

from utils.progress import Progress
from utils.logger import Logger

from benchmarks.classic import acer_parts
import metrics

from evaluation import evaluation_classic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="ACER")
    parser.add_argument("--comment", default="supsup")
    parser.add_argument("--namestr", default="ACER")
    parser.add_argument("--env_name", default="LunarLander-v2", help="CartPole-v0, LunarLander-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--eval_freq", default=1000, type=float)
    parser.add_argument("--entropy_weight", default=0.001, type=float)
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--target_freq", default=100, type=int)
    parser.add_argument("--k_steps", default=5, type=int)
    parser.add_argument("--max_eps_length", default=500, type=int)
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--gamma", default=0.99, type=float,
                        help="discount factor.")
    parser.add_argument("--steps_per_bar", default=500, type=int,
        help='Steps per progress bar')
    parser.add_argument('--checkpoint_dir', type=str, default='',
                        help='chkpt directory name')
    parser.add_argument("--var_buff_len", default=10, type=int, \
            help="Maximum Buffer Size for moving variance calculation")
    parser.add_argument("--log_tb", action="store_true", default=False,
            help='Log with tensorboard')
    parser.add_argument("--log_comet", action="store_true", default=False,
            help='Use comet for logging')
    args = parser.parse_args()

    env_id = args.env_name
    env = gym.make(env_id)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pbar = Progress(args.steps_per_bar) # nice progress bar
    evaluator = evaluation_classic.Evaluator(
            env,
            wrap_function=evaluation_classic.wrap_acer_policy,
            n_episodes=10)

    if args.log_comet or args.log_tb:
        args.log = True

    # Logging
    if args.log:
        logger = Logger(args, "classic_acer_logs/", args.checkpoint_dir)

    model = acer_parts.ActorCritic(
        env.observation_space.shape[0],
        env.action_space.n).to(device)

    optimizer = optim.Adam(
            model.parameters(),
            eps=1e-3)
    #optimizer = optim.SGD(model.parameters(), lr=1e-3)

    capacity = 1000000
    max_episode_length = args.max_eps_length
    replay_buffer = acer_parts.EpisodicReplayMemory(
        capacity, max_episode_length)

    total_timesteps = 0
    num_steps = args.k_steps

    all_rewards = []
    q_grad_queue = deque([])
    actor_grad_queue = deque([])
    state = env.reset()
    total_episodes = 0

    evaluations = [evaluator.evaluate_policy(model, pbar)]

    running_average_rewards = None

    print("Environment :", args.env_name)
    episode_reward = 0
    all_rewards = []
    while total_timesteps < args.max_timesteps:

        q_values = []
        values = []
        policies = []
        actions = []
        rewards = []
        masks = []

        if total_timesteps % args.steps_per_bar == 0:
            pbar.epoch_start()


        for step in range(num_steps):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            policy, q_value, value = model(state)
            action = policy.multinomial(1)
            next_state, reward, done, _ = env.step(action.cpu().item())
            episode_reward += reward

            reward = torch.FloatTensor([reward]).unsqueeze(1).to(device)
            mask = torch.FloatTensor(1 - np.float32([done])).unsqueeze(1).to(device)
            replay_buffer.push(
                state.detach(), action, reward, policy.detach(), mask, done)

            q_values.append(q_value)
            policies.append(policy)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            masks.append(mask)

            state = next_state

            if done:
                if running_average_rewards is None:
                    running_average_rewards = episode_reward
                else:
                    running_average_rewards = (
                        0.9 * running_average_rewards + 0.1 * episode_reward
                    )
                                
                pbar.print_train(
                    total_episodes,
                    total_timesteps,
                    episode_reward,
                    running_average_rewards, 0)

                if args.log:
                    logger.log("Train Reward",episode_reward,step=total_timesteps)
                    grad_norm = metrics.grad_norm(model.critic)
                    logger.log("Q Net Grad Norm",grad_norm,step=total_timesteps)
                    weight_norm = metrics.weight_norm(model.critic)
                    logger.log("Q Net Weight Norm",weight_norm,step=total_timesteps)
                    grad_var = metrics.grad_var(q_grad_queue,grad_norm,args.var_buff_len)
                    logger.log("Q Net Grad Var",grad_var,step=total_timesteps)
                
                    grad_norm = metrics.grad_norm(model.actor)
                    logger.log("Actor Net Grad Norm",grad_norm,step=total_timesteps)
                    weight_norm = metrics.weight_norm(model.actor)
                    logger.log("Actor Net Weight Norm",weight_norm,step=total_timesteps)
                    grad_var = metrics.grad_var(actor_grad_queue,grad_norm,args.var_buff_len)
                    logger.log("Actor Net Grad Var",grad_var,step=total_timesteps)
                state = env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0
                total_episodes += 1
                env.reset()  # TODO(riashat): Check
                done = False

        next_state = torch.FloatTensor(state).unsqueeze(0).to(device)
        _, _, retrace = model(next_state)
        retrace = retrace.detach()
        loss = acer_parts.compute_acer_loss(policies, q_values, values,
                                            actions, rewards, retrace,
                                            masks, policies, gamma=args.gamma, entropy_weight=args.entropy_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # This function will update the parameters using
        # the off policy loss.
        acer_parts.off_policy_update(
                replay_buffer,
                model,
                args.batch_size,
                optimizer,
                gamma=args.gamma,
                entropy_weight=args.entropy_weight)



        if total_timesteps % args.eval_freq == 0:
            evaluations.append(evaluator.evaluate_policy(model, pbar))

            if args.log:
                logger.log('Eval Reward', evaluations[-1], step=total_timesteps)
                #if args.save_models:
                #    torch.save(model, os.path.join('pytorch_models', file_name))
                #    np.save("./results/%s" % (file_name), evaluations)


        total_timesteps += num_steps

