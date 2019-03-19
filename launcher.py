#CUDA_VISIBLE_DEVICES=0 python launcher.py --policy_name MaxEntDDPG --use_logger True --beta 0.8 --env_name InvertedPendulum-v1
import os
import itertools
import numpy as np
import subprocess
import argparse
parser = argparse.ArgumentParser()

def grid_search(args_vals):
    """ arg_vals: a list of lists, each one of format (argument, list of possible values) """
    lists = []
    for arg_vals in args_vals:
        arg, vals = arg_vals
        ll = []
        for val in vals:
            ll.append("-" + arg + " " + str(val) + " ")
        lists.append(ll)
    return ["".join(item) for item in itertools.product(*lists)]


parser = argparse.ArgumentParser()
parser.add_argument('--experiments', type=int, default=10)
parser.add_argument('--policy_name', type=str, default="DDPG")          # Policy name
parser.add_argument('--env_name', type=str, default="HalfCheetah-v1")         # OpenAI gym environment name
parser.add_argument('--start_timesteps', default=10000, type=int)     # How many time steps purely random policy is run for
parser.add_argument('--eval_freq', default=5e3, type=float)         # How often (time steps) we evaluate
parser.add_argument('--max_timesteps', default=1e6, type=float)     # Max time steps to run environment for
parser.add_argument('--save_models', default=True,type=bool)           # Whether or not models are saved
parser.add_argument('--expl_noise', default=0.1, type=float)        # Std of Gaussian exploration noise
parser.add_argument('--batch_size', default=100, type=int)          # Batch size for both actor and critic
parser.add_argument('--discount', default=0.99, type=float)         # Discount factor
parser.add_argument('--tau', default=0.005, type=float)             # Target network update rate
parser.add_argument('--policy_noise', default=0.2, type=float)      # Noise added to target policy during critic update
parser.add_argument('--noise_clip', default=0.5, type=float)        # Range to clip target policy noise
parser.add_argument('--policy_freq', default=2, type=int)           # Frequency of delayed policy updates
parser.add_argument("--ent_weight", default=0.005, type=float)
parser.add_argument('-g',  type=str, default='0', help=['specify GPU'])
parser.add_argument('--folder', type=str, default="./results/")          # Folder to save results in
parser.add_argument("--use_logger", type=bool, default=False, help='whether to use logging or not')
parser.add_argument("--no_new_samples_after_threshold", type=bool, default=False, help='stop adding new samples to the replay buffer')
parser.add_argument("--add_buffer_threshold", type=int, default=2, help='threshold after which to add samples to buffer') ## stop adding new samples to the buffer

parser.add_argument("--action_interpolation", type=bool, default=False, help='interpolate between on-policy and off-policy actions')
parser.add_argument("--beta", type=float, default=1.0, help='parameter controlling interpolation between on-policy and off-policy actions')
parser.add_argument("--control_buffer_samples", type=bool, default=False, help='control when to add samples to the buffer')
parser.add_argument("--repeated_critic_updates", type=bool, default=False, help='do repeated updates of the critic')
parser.add_argument("--critic_repeat", type=float, default=5, help='number of repeated updates of the critic')
parser.add_argument("--on_policy", type=bool, default=False, help='Be completely on-policy')
parser.add_argument("--off_policy", type=bool, default=False, help='Be completely off-policy')
parser.add_argument("--larger_critic_approximator", type=bool, default=False, help='Use a higher capacity function approximator for the critic')

parser.add_argument('--main', type=str, default="./main.py")   



locals().update(parser.parse_args().__dict__)    


job_prefix = "python "
# exp_script = './main.py '
# job_prefix += exp_script

args = parser.parse_args()
exp_script = args.main + ' ' 
job_prefix += exp_script


experiments = args.experiments
policy_name = args.policy_name
env_name = args.env_name
start_timesteps = args.start_timesteps
eval_freq = args.eval_freq
max_timesteps = args.max_timesteps
expl_noise = args.expl_noise
batch_size = args.batch_size
discount = args.discount
tau = args.tau
policy_noise = args.policy_noise
noise_clip = args.noise_clip
policy_freq = args.policy_freq
ent_weight = args.ent_weight
folder = args.folder
save_models = args.save_models
no_new_samples_after_threshold = args.no_new_samples_after_threshold
add_buffer_threshold = args.add_buffer_threshold
action_interpolation = args.action_interpolation
control_buffer_samples = args.control_buffer_samples
repeated_critic_updates = args.repeated_critic_updates
critic_repeat = args.critic_repeat
on_policy = args.on_policy
off_policy = args.off_policy
larger_critic_approximator = args.larger_critic_approximator
use_logger = args.use_logger
beta = args.beta

grid = [] 
grid += [['-policy_name', [policy_name]]]
grid += [['-env_name', [env_name]]]
grid += [['-start_timesteps', [start_timesteps]]]
grid += [['-eval_freq', [eval_freq]]]
grid += [['-max_timesteps', [max_timesteps]]]
grid += [['-save_models',[args.save_models]]]
grid += [['-expl_noise', [expl_noise]]]
grid += [['-batch_size', [batch_size]]]
grid += [['-batch_size', [batch_size]]]
grid += [['-tau', [tau]]]
grid += [['-policy_noise', [policy_noise]]]
grid += [['-noise_clip', [noise_clip]]]
grid += [['-policy_freq', [policy_freq]]]
grid += [['-ent_weight', [ent_weight]]]
grid += [['-folder', [folder]]]
grid += [['-use_logger', [use_logger]]]
grid += [['-beta', [beta]]]
grid += [['-no_new_samples_after_threshold', [no_new_samples_after_threshold]]]
grid += [['-add_buffer_threshold', [add_buffer_threshold]]]
grid += [['-action_interpolation', [action_interpolation]]]
grid += [['-control_buffer_samples', [control_buffer_samples]]]
grid += [['-repeated_critic_updates', [repeated_critic_updates]]]
grid += [['-critic_repeat', [critic_repeat]]]
grid += [['-on_policy', [on_policy]]]
grid += [['-off_policy', [off_policy]]]
grid += [['-larger_critic_approximator', [larger_critic_approximator]]]


job_strs = []
for settings in grid_search(grid):
    for e in range(experiments):    
        job_str = job_prefix + settings
        job_strs.append(job_str)
print("njobs", len(job_strs))

for job_str in job_strs:
    os.system(job_str)
