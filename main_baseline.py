#from comet_ml import Experiment
import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3
import OurDDPG
import DDPG
from utils import Logger
from utils import create_folder
from plot import main_plot
import datetime
from tensorboardX import SummaryWriter

# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print ("---------------------------------------")
    print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print ("---------------------------------------")
    return avg_reward


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="DDPG")                    # Policy name
    parser.add_argument("--env_name", default="HalfCheetah-v1")         # OpenAI gym environment name
    parser.add_argument("--seed", default=-1, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)     # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)         # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)     # Max time steps to run environment for
    parser.add_argument("--save_models", default=True)          # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)        # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)          # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)         # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)             # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)      # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)        # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)           # Frequency of delayed policy updates
    parser.add_argument("--ent_weight", default=0.01, type=float)       # Range to clip target policy noise
    parser.add_argument("--folder", type=str, default='./results/')
    parser.add_argument("--name_prefix", type=str, default="")

    parser.add_argument("--use_logger", type=bool, default=False, help='whether to use logging or not')
    parser.add_argument("--no_new_samples_after_threshold", type=bool, default=False, help='stop adding new samples to the replay buffer')
    parser.add_argument("--add_buffer_threshold", type=int, default=2, help='threshold after which to add samples to buffer') ## stop adding new samples to the buffer
    parser.add_argument("--action_interpolation", type=bool, default=False, help='interpolate between on-policy and off-policy actions')
    parser.add_argument("--beta", type=float, default=1.0, help='parameter controlling interpolation between on-policy and off-policy actions')
    parser.add_argument("--control_buffer_samples", type=bool, default=False, help='control when to add samples to the buffer')
    parser.add_argument("--repeated_critic_updates", type=bool, default=False, help='do repeated updates of the critic')
    parser.add_argument("--critic_repeat", type=float, default=1, help='number of repeated updates of the critic')
    parser.add_argument("--on_policy", type=bool, default=False, help='Be completely on-policy')
    parser.add_argument("--off_policy", type=bool, default=False, help='Be completely off-policy')
    parser.add_argument("--larger_critic_approximator", type=bool, default=False, help='Use a higher capacity function approximator for the critic')
    parser.add_argument("--exp-name", type=str, default="exp")

    args = parser.parse_args()

    if args.seed == -1:
        args.seed = np.random.randint(10)

    if args.use_logger:
        file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))

        logger = Logger(experiment_name = args.policy_name, environment_name = args.env_name, folder = args.folder)
        logger.save_args(args)

        print ('Saving to', logger.save_folder)


    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    env = gym.make(args.env_name)

    # Set seeds
    seed = args.seed
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.use_logger: 
        print ("---------------------------------------")
        print ("Settings: %s" % (file_name))
        print ("Seed : %s" % (seed))
        print ("---------------------------------------")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    #cometml = Experiment(api_key="Gncqbz3Rhfy3MZJBcX7xKVJoo", project_name="rl-767", workspace="deepak-sharma-mail-mcgill-ca")
    #cometml.log_parameters(vars(args))
    #cometml.set_name(f"{args.name_prefix}_{args.policy_name}_{args.env_name}_{args.seed}_{args.batch_size}_{args.critic_repeat}_{args.larger_critic_approximator}")
    #cometml.add_tags([args.name_prefix, args.policy_name, args.env_name])

    date_of_exp = datetime.datetime.now().strftime("%d-%m-%y-%H-%M-%S")
    writer = SummaryWriter(f"runs/{args.name_prefix}_{args.policy_name}_{args.env_name}_{args.seed}_{args.batch_size}_{args.critic_repeat}_{args.larger_critic_approximator}_{date_of_exp}")

    # Initialize policy
    if args.policy_name == "TD3": policy = TD3.TD3(state_dim, action_dim, max_action)
    elif args.policy_name == "DDPG": policy = DDPG.DDPG(state_dim, action_dim, max_action, args.larger_critic_approximator)

    replay_buffer = utils.ReplayBuffer()
    
    # Evaluate untrained policy
    evaluations = [evaluate_policy(policy)] 
    episode_reward = 0 
    training_evaluations = [episode_reward]

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    eval_ctr = 1
    done = True 
    
    while total_timesteps < args.max_timesteps:
        
        if done: 

            if total_timesteps != 0: 
                print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (total_timesteps, episode_num, episode_timesteps, episode_reward))
                #with cometml.train():
                if args.policy_name == "TD3":
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq)
                else: 
                    actor_losses, critic_losses = policy.train(replay_buffer, episode_timesteps, args.repeated_critic_updates, args.critic_repeat, args.batch_size, args.discount, args.tau)
                    act_loss = float(torch.mean(torch.stack(actor_losses)))
                    crt_loss = float(torch.mean(torch.stack(critic_losses)))
                    
#                    cometml.log_metric("actor_loss", act_loss, episode_num)
#                    cometml.log_metric("critic_loss", crt_loss, episode_num)
#                    cometml.log_metric("return", float(episode_reward), episode_num)

                    writer.add_scalar("train_actor_loss", act_loss, episode_num)
                    writer.add_scalar("train_critic_loss", crt_loss, episode_num)
                    writer.add_scalar("train_return", float(episode_reward), episode_num)

                    if args.use_logger:
                        logger.log_metric("actor_losses", act_loss)
                        logger.log_metric("critic_losses", crt_loss)

            
            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                #with cometml.validate():
                timesteps_since_eval %= args.eval_freq
                ret = evaluate_policy(policy)
                evaluations.append(ret)
                #cometml.log_metric("return", float(ret), eval_ctr)
                writer.add_scalar("valid_return", float(ret), eval_ctr)
                eval_ctr += 1
                if args.use_logger:
                    logger.record_reward(evaluations)
                    logger.save()           
                    if args.save_models: policy.save(file_name, directory="./pytorch_models")
                    np.save("./results/%s" % (file_name), evaluations) 
            
            # Reset environment
            obs = env.reset()
            done = False
            training_evaluations.append(episode_reward)

            if args.use_logger:
                logger.training_record_reward(training_evaluations)
                logger.save_2()
                
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
        
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(obs))
            action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)              


        # Perform action
        new_obs, reward, done, _ = env.step(action) 
        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        episode_reward += reward
        
        replay_buffer.add((obs, new_obs, action, reward, done_bool))
        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
        
    # Final evaluation 
   # with cometml.validate():
    evaluations.append(evaluate_policy(policy))
   # cometml.log_metric("return", evaluations[-1], eval_ctr)
    writer.add_scalar("valid_return", evaluations[-1], eval_ctr)
    
    #with cometml.train():
    training_evaluations.append(episode_reward)
    #    cometml.log_metric("return", episode_reward, episode_num)
    writer.add_scalar("train_return", episode_reward, episode_num)

    if args.use_logger:
        logger.record_reward(evaluations)
        logger.training_record_reward(training_evaluations)
        logger.save()
        logger.save_2()

        act_fig = main_plot([np.array(logger.get_metric("actor_losses"))], labels = ['actor_loss'], file_name = logger.save_folder + "/actor_loss", title = "Actor Loss vs Episode", x_label = "Episodes", y_label = "Loss")

        crt_fig = main_plot([np.array(logger.get_metric("critic_losses"))], labels = ['critic_loss'], file_name = logger.save_folder + "/critic_loss", title = "Critic Loss vs Episode", x_label = "Episodes", y_label = "Loss")

        training_fig = main_plot([np.array(training_evaluations)], labels = ['Return during training'], file_name = logger.save_folder + "/training", title = "Train Return vs Iteration", x_label = "Iterations", y_label = "Rewards")

        validation_fig = main_plot([np.array(evaluations)], labels = ['Reward during evaulation'], file_name = logger.save_folder + "/evaluation", title = "Eval Return vs Iteration", x_label = "Iterations", y_label = "Rewards")

        #cometml.log_figure(figure_name = "Actor Loss vs Episode", figure = act_fig)
        #cometml.log_figure(figure_name = "Critic Loss vs Episode", figure = crt_fig)
        #cometml.log_figure(figure_name = "Eval return vs Iteration", figure = validation_fig)
        #cometml.log_figure(figure_name = "Train return vs Iteration", figure = training_fig)
        
        writer.add_figure(tag = "Actor Loss vs Episode", figure = act_fig)
        writer.add_figure(tag = "Critic Loss vs Episode", figure = crt_fig)
        writer.add_figure(tag = "Eval return vs Iteration", figure = validation_fig)
        writer.add_figure(tag = "Train return vs Iteration", figure = training_fig)
        
        if args.save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
        np.save("./results/%s" % (file_name), evaluations)  
