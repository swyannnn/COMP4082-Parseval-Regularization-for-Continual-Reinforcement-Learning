import sys, os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch 
torch.set_num_threads(8)       # number of physical cores you want
torch.set_num_interop_threads(8)
from torch.utils.tensorboard import SummaryWriter
import pickle
from collections import defaultdict
import numpy as np
import gymnasium as gym
import subprocess
import argparse
import time

def get_device():
    """
    Select available device (CPU/GPU/MPS) with the most free memory.
    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        try:
            # Query nvidia-smi for free memory per GPU
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                encoding='utf-8', capture_output=True, check=True
            )
            # Parse into list of ints
            free_mem = [int(x) for x in result.stdout.strip().split('\n')]
            best_gpu = int(np.argmax(free_mem))
            print(f"[GPU SELECTOR] Choosing GPU {best_gpu} with {free_mem[best_gpu]} MB free.")
            device = torch.device(f"cuda:{best_gpu}")
        except Exception as e:
            print("Could not query nvidia-smi, defaulting to GPU 0.", e)
            device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    return device

class ConfigDictConverter:
    def __init__(self, config_dict):
        '''
        This class takes a config_dict which contains all the variables needed to do one run
        and converts it into the variables needed to run the RL experiments
        We assume that the config file has certain variables and is organized in the proper way
        For the env and agent parameters, we will pass config_dict on to them and assume that they handle it properly
        Note that we *cannot* use the same variable names for env and agent parameters. If the agent or env expect the
        same name, we will need to write it down different in the config file and then convert here

        Attributes:
        agent_dict:
        env_dict:
        '''
        # Improvement: possible to split agent and env parameters here. That is this class contains
        # two dicts for agent_parameters and env_parameters.
        # This would help with passing only the required arguments for envs that are already created (e.g gym)
        # Also, it could help with dealing with parameters that have the same name but are different for agent and env

        self.config_dict = config_dict.copy()

        # training shouldn't need these variables
        if 'num_repeats' in self.config_dict.keys():
            del self.config_dict['num_repeats']
        if 'num_runs_per_group' in self.config_dict.keys():
            del self.config_dict['num_runs_per_group']

        self.agent_dict = self.config_dict.copy()
        self.env_dict = self.config_dict.copy()

        # TODO remove maybe?
        self.repeat_idx = config_dict['repeat_idx']  

        # algorithm for RL
        if config_dict['base_algorithm'].lower() in ('ppo_agent',):
            agent = config_dict['base_algorithm'].lower()

            if agent == 'ppo_agent':
                from agent import PPO_Agent

                self.agent_class = PPO_Agent
                agent_key_lst = [
                                'base_algorithm', 'device', 'learning_rate', 'num_envs', 'rollout_num_steps', 'anneal_lr', 'gamma', 'gae_lambda', 'num_minibatches', 'minibatch_size', 'update_epochs', 'norm_adv', 'clip_coef', 'clip_vloss', 'ent_coef', 'vf_coef', 'max_grad_norm', 'target_kl',
                                'layer_norm', 'layer_norm_no_params', 'weight_decay', 'tuned_adam', 'parseval_reg', 'parseval_num_groups',
                                'perturb', 'perturb_dist',
                                'weight_init', 'net_width', 'net_activation','init_gain',
                                'adam_eps', 'adam_beta2', 'tsallis_entropy', 'l2_init', 'group_sort',
                                ]
        else:
            raise AssertionError('Invalid algorithm', config_dict['base_algorithm'])

        self.agent_dict = {k: v for k, v in self.agent_dict.items() if k in agent_key_lst}
        print('agent_dict keys', self.agent_dict.keys())
        
        if 'gym_continual_pendulum' in config_dict['env'].lower():
            from envs.pendulum_env import ContinualPendulumSequence
            self.env_class = lambda **kwargs: ContinualPendulumSequence()
            self.env_dict = {'env': 'gym_continual_pendulum', 'env_type': 'rl', 'seed': config_dict.get('seed', 123)}

        elif 'gym_continuous_drift_pendulum' in config_dict['env'].lower():
            from envs.pendulum_env import ContinuousDriftPendulum
            self.env_class = lambda **kwargs: ContinuousDriftPendulum(
                gym.make("Pendulum-v1", render_mode="rgb_array"),
                param_name="gravity",
                task_values=(5.0, 10.0, 15.0),
                change_freq=50000
            )
            self.env_dict = {'env': 'gym_continuous_drift_pendulum', 'env_type': 'rl', 'seed': config_dict.get('seed', 123)}

        else:
            raise AssertionError("config dict converter: env doesn't match" + config_dict['env'])

        print('env_dict keys', self.env_dict.keys())

        # Other params
        self.agent_dict['device'] = self.config_dict['device']
        print("ConfigDictConverter: agent device", self.agent_dict['device'])

        # Adjust the seed based on repeat
        self.env_dict['seed'] += self.repeat_idx*1


        # local run
        if 'local_run' in self.config_dict and self.config_dict['local_run']:
            self.env_dict['local_run'] = True
        else:
            self.env_dict['local_run'] = False


class RLLogger:
    def __init__(self, save_freq, save_model_freq=None, config_idx=0):
        self.metrics = defaultdict(list)
        self.save_freq = save_freq
        self.save_model_freq = save_model_freq  # how often to save the model checkpoints (less frequent usually)

    def save_metrics(self, option, agent=None, loss=None, episode_return=None, save_path="", save_tag="", *args,
                     **kwargs):
        os.makedirs(os.path.join(save_path, 'models'), exist_ok=True)
        # save model
        if option == 'standard':
            self.metrics['loss'].append(loss)
            for k, v in kwargs.items():
                self.metrics[k].append(v)
        elif option == 'episode':
            self.metrics['return'].append(episode_return)
            for k, v in kwargs.items():
                self.metrics[k].append(v)
        elif option == 'eval':
            # eval_return
            for k, v in kwargs.items():
                self.metrics[k].append(v)
        elif option == 'model':
            agent.save_model(os.path.join(save_path, 'models', f'model_{save_tag}_{kwargs.get("total_num_steps")}.pyt'))
            print('saved agent')
        elif option == 'states':
            states = kwargs['states']
            torch.save(states, os.path.join(save_path, 'models', f'states_{save_tag}_{kwargs.get("total_num_steps")}.pt'))
            print('saved states')

    def reset(self):
        self.metrics = defaultdict(list)

    def save_to_file(self, save_path, save_tag):
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"data_{save_tag}.pkl")

        # np.save(save_path + f'data_{save_tag}.npy', self.metrics)
        with open(file_path, 'wb') as file:
            pickle.dump(self.metrics, file, protocol=pickle.HIGHEST_PROTOCOL)

    def plot_summary(self):
        ''' Prints averages of the metrics '''
        for k, v in self.metrics.items():
            print(f"{k}: mean {np.mean(v)}, std {np.std(v)}")



# python parseval_reg/parseval_reg/main_modified.py --env gym_continual_pendulum --algorithm base --learnable_input_scale True --add_diag_layer True --drift_rate 2e-5 --net_width 64 --net_activation tanh --weight_init orthogonal --repeat_idx 1 --num_steps 10000000
def main():
    parser = argparse.ArgumentParser(description='Run RL experiments')
    parser.add_argument('--test_run', action='store_true', help='Run a test run with only 10k steps and eval/save every 2k steps')

    # General experiment arguments
    parser.add_argument('--algorithm', type=str, default='parseval', help='Learning algorithm to run for the experiment')
    parser.add_argument('--base_algorithm', type=str, default='ppo_agent', help='Base algorithm for the agent')
    parser.add_argument('--repeat_idx', type=int, default=0, help='Index of the repeat (for multiple runs)')
    parser.add_argument('--env', type=str, default='gym_pendulum', help='Environment to run')
    parser.add_argument('--drift_rate', type=float, default=2e-5, help='Rate of parameter drift per step')
    parser.add_argument('--drift_param', type=str, default='gravity', help='Which parameter to drift')
    parser.add_argument('--change_freq', type=int, default=1e6, help='Frequency to change tasks in the environment')
    parser.add_argument('--num_steps', type=int, default=10000, help='Num steps to run' )
    parser.add_argument('--seed', type=int, default=123, help='Num steps to run')
    parser.add_argument('--save_path', type=str, default='results1/', help='Path to the folder to be saved in')
    parser.add_argument('--save_freq', type=int, default=25000, help='Number steps between recording metrics')
    parser.add_argument('--save_model_freq', type=int, default=100000, help='Number of steps between saving the model. Set to -1 for never. ')
    parser.add_argument('--render', action='store_true', help='Render the environment during training')
    

    # Arguments for PPO/RPO
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--num_envs', type=int, default=1, help='Number of environments')
    parser.add_argument('--rollout_num_steps', type=int, default=2048, help='Number of steps per rollout')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor (gamma)')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--num_minibatches', type=int, default=32, help='Number of minibatches')
    parser.add_argument('--update_epochs', type=int, default=10, help='Number of epochs per update')
    parser.add_argument('--minibatch_size', type=int, default=64, help='Size of each minibatch')
    parser.add_argument('--norm_adv', type=bool, default=True, help='Normalize advantages')
    parser.add_argument('--clip_coef', type=float, default=0.2, help='Clip coefficient')
    parser.add_argument('--clip_vloss', type=bool, default=True, help='Clip value loss')
    parser.add_argument('--ent_coef', type=float, default=0.0, help='Entropy coefficient')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Maximum gradient norm')
    parser.add_argument('--rpo_alpha', type=float, default=0.5, help='RPO alpha parameter')

    # Parseval regularization
    parser.add_argument('--parseval_reg', type=float, default=0, help='Parseval regularization coefficient')

    # Adding additional learnable parameters
    parser.add_argument('--input_scale', type=float, default=1, help='Input scaling factor')
    parser.add_argument('--learnable_input_scale', type=bool, default=False, help='Make input scale learnable')
    parser.add_argument('--add_diag_layer', type=bool, default=False, help='Add diagonal layers to the network')

    # Other loss of plasticity methods
    parser.add_argument('--layer_norm', type=bool, default=False, help='Use layer normalization')
    parser.add_argument('--layer_norm_no_params', type=bool, default=False, help='Layer norm without additional parameters')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay coefficient')
    parser.add_argument('--tuned_adam', type=bool, default=False, help='Use tuned Adam optimizer')
    # Perturb (for Shrink-and-Perturb, combine with weight decay)
    parser.add_argument('--perturb', type=float, default=0.0, help='Perturbation factor')
    parser.add_argument('--perturb_dist', type=str, default='xavier', help='Distribution for perturbations (e.g., xavier)')
    # Regenerative regularization
    parser.add_argument('--regen', type=float, default=0.0, help='Regeneration factor')
    parser.add_argument('--regen_wasserstein', type=bool, default=False, help='Use Wasserstein loss for regeneration')

    # Network architecture arguments
    parser.add_argument('--weight_init', type=str, default='orthogonal', help='Weight initialization method')
    parser.add_argument('--net_width', type=int, default=64, help='Width of the network layers')
    parser.add_argument('--net_activation', type=str, default='tanh', help='Activation function for the network')
    parser.add_argument('--init_gain', type=float, default=None, help='Gain for weight initialization (if applicable)')

    # Ablations
    parser.add_argument('--parseval_norm', type=bool, default=False, help='Normalize row weight vectors before applying Parseval reg')
    parser.add_argument('--parseval_num_groups', type=int, default=1, help='Number of groups for Parseval regularization')


    args = parser.parse_args()
    
    # Set different defaults depending on the env and algorithm
    if "gym_" in args.env:
        # ✅ default hyperparameters for continuous Gym control environments (Pendulum, MountainCar, etc.)
        args.num_steps = 1e7
        args.change_freq = args.num_steps//8 

        # PPO training defaults
        args.rollout_num_steps = 2048  
        args.num_minibatches = 32  
        args.update_epochs = 10  
        args.minibatch_size = 64  
        args.ent_coef = 0.01
        args.rpo_alpha = 0
        args.learning_rate = 0.00025

        # Parseval and EWC hyperparameters
        if args.algorithm == 'base':
            pass
        elif args.algorithm == 'parseval':
            args.parseval_reg = 0.001
        elif args.algorithm == 'ewc':
            args.ewc_lambda = 1000
        elif args.algorithm == 'layer_norm':
            args.layer_norm = True
        elif args.algorithm == 'snp':
            args.perturb = 0.001
            args.weight_decay = 0.001
        elif args.algorithm == 'regen':
            args.regen = 0.001
        elif args.algorithm == 'w-regen':
            args.regen = 0.001
            args.regen_wasserstein = True

    else:
        raise AssertionError("Invalid env", args.env)

    args.device = get_device()
    # time.sleep(5)  # wait for a bit to avoid potential GPU allocation issues

    # initialize config
    save_tag = f"{args.env}_{args.algorithm}_{args.repeat_idx}"
    config_obj = ConfigDictConverter(vars(args))
    agent_parameters = config_obj.agent_dict
    env_parameters = config_obj.env_dict
    num_steps_per_run = args.num_steps

    print(f"Start RL run!")
    print(agent_parameters)
    print(env_parameters)

    start_time = time.perf_counter()

    # initialize logger, agent, env
    metric_logger = RLLogger(args.save_freq, args.save_model_freq)
    metric_logger.reset()  # reset for run

    # initialize TensorBoard writer
    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(log_dir=f"{args.save_path}/{save_tag}")

    env = config_obj.env_class(**env_parameters)

    agent_parameters.update({"env": env})  # use 'env' to pass to agent
    agent_parameters.update({"device": args.device})
    agent = config_obj.agent_class(**agent_parameters)
 
    obs, _ = env.reset()  # match the gymnasium interface

    i_step = 0

    if metric_logger.save_model_freq > 0:  # there's an option not to save models
        metric_logger.save_metrics(option='model',
                                        agent=agent,
                                        total_num_steps=i_step,
                                        save_path=args.save_path,
                                        save_tag=save_tag
                                        )

    while i_step < num_steps_per_run:
        actions = agent.act(obs)
        next_obs, rewards, terminateds, truncateds, infos = env.step(actions)

        # also updates parameters
        agent.update(obs, next_obs, actions, rewards, terminateds, truncateds, infos)

        obs = next_obs

        ## For single (non-vector) envs
        if isinstance(terminateds, bool):  # check that it is a single env
            if "episode" in infos:
                if env_parameters['local_run']:
                    print(f"global_step={i_step}, episodic_return={infos['episode']['r']}")

                temp_online_return = infos['episode']['r']
                metric_logger.save_metrics('episode', episode_return=temp_online_return, step=i_step, save_path=args.save_path)
                writer.add_scalar("episode/return", temp_online_return, i_step)

            if terminateds or truncateds:
                obs, _ = env.reset()

        if i_step == 0 or (i_step + 1) % metric_logger.save_freq == 0:
            print('time: ', round((time.perf_counter() - start_time)/60,3), "SPS:", int(i_step / (time.perf_counter() - start_time)))

            save_metrics = {}
            if agent_parameters['base_algorithm'] == 'ppo_agent':
                if agent.explained_var is not None:
                    save_metrics['explained_var'] = agent.explained_var
                    save_metrics['entropy'] = agent.entropy
                    logged_values = agent.get_log_quantities()
                    save_metrics.update(logged_values)
                    for k, v in save_metrics.items():
                        if isinstance(v, (float, int, np.floating, np.integer)):
                            writer.add_scalar(f"train/{k}", v, i_step)
                metric_logger.save_metrics('standard', **save_metrics, save_path=args.save_path)

        render = args.render
        if i_step == 0 or (i_step + 1) % metric_logger.save_freq == 0:  # eval step
            save_metrics = {}
            # eval
            num_eval_runs = 10
            eval_results = env.evaluate_agent(agent, num_eval_runs, render=render)
            print("gravity_values")
            eval_episode_returns = eval_results['episodic_returns']
            save_metrics['runtime'] = (time.perf_counter() - start_time) / 60  # time in mins

            save_metrics['mean_eval_return'] = np.mean(eval_episode_returns)
            save_metrics['std_eval_return'] = np.std(eval_episode_returns, ddof=1)
            save_metrics['min_eval_return'] = np.min(eval_episode_returns)
            save_metrics['max_eval_return'] = np.max(eval_episode_returns)

            if "successes" in eval_results.keys():
                eval_successes = eval_results['successes']
                save_metrics['mean_eval_success'] = np.mean(eval_successes)
                save_metrics['std_eval_success'] = np.std(eval_successes, ddof=1)
                writer.add_scalar("eval/mean_success", save_metrics['mean_eval_success'], i_step)
                writer.add_scalar("eval/std_success", save_metrics['std_eval_success'], i_step)
                print(f"{i_step} success {round(save_metrics['mean_eval_success'],3)} +/- {round(save_metrics['std_eval_success']/np.sqrt(num_eval_runs),3)}")

            writer.add_scalar("eval/mean_return", save_metrics['mean_eval_return'], i_step)
            writer.add_scalar("eval/std_return", save_metrics['std_eval_return'], i_step)
            writer.add_scalar("eval/min_return", save_metrics['min_eval_return'], i_step)
            writer.add_scalar("eval/max_return", save_metrics['max_eval_return'], i_step)
            print(f"{i_step} eval return {round(save_metrics['mean_eval_return'],3)} +/- {round(save_metrics['std_eval_return']/np.sqrt(num_eval_runs),3)}")
            metric_logger.save_metrics('eval', **save_metrics, save_path=args.save_path)

        if metric_logger.save_model_freq > 0:  # there's an option not to save models
            if (i_step+1) % metric_logger.save_model_freq == 0:
                metric_logger.save_metrics(option='model',
                                                agent=agent,
                                                total_num_steps=(i_step+1),
                                                save_path=args.save_path,
                                                save_tag=save_tag)
                if agent_parameters['base_algorithm'] == 'ppo_agent':  # save states for checking later
                    metric_logger.save_metrics(option='states', agent=agent,
                                                    total_num_steps=(i_step+1),
                                                    states=agent.obs.clone().detach(),
                                                    save_path=args.save_path, 
                                                    save_tag=save_tag)

        sys.stdout.flush()
        sys.stderr.flush()

        i_step += 1


    # save data
    metric_logger.save_to_file(args.save_path, save_tag=save_tag)


    print('done RL run. Time {} min'.format((time.perf_counter() - start_time)/60))

    sys.stdout.flush()
    sys.stderr.flush()
    writer.close()
    return metric_logger



if __name__ == "__main__":
    main()