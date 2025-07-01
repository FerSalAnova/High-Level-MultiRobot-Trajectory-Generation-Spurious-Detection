from abc import ABC, abstractmethod
from enum import Enum
import torch
from torch.autograd import Variable
import numpy as np
from Task.Task import Task
from vmas import make_env
import traceback

class TaskVMAS(Task):

    def __init__(self, config, agent_input_size, communication_radius):     
        self.action_dim_per_agent = 2
        self.obs_dim_per_agent = agent_input_size
   
        super().__init__(config, agent_input_size, communication_radius)

        self.vmas_scenario =  config["task"]["type"] 
        self.__build_real_dynamics()
        
    def getVMASConfig(self, expert_robots=False):
        return {}

    
    def buildFeatureIndex(self):
        pos_idx = []
        vel_idx = []
        for i in range(0, self.numAgents*self.obs_dim_per_agent, self.obs_dim_per_agent):
            pos_idx.append(i)
            pos_idx.append(i+1)
            vel_idx.append(i+2)
            vel_idx.append(i+3)

        feature_index = {
            "robot_positions": pos_idx,
            "robot_velocities": vel_idx,
        }   
        return feature_index   

    def buildInputVariables(self, inputs):
        na = self.numAgents
        i = int(inputs.shape[1] / na)
        
        return Variable(inputs.reshape(-1, na, i).transpose(1, 2).data, requires_grad=True)
    
    def reshapeObservation(self, obs):
        batch_size = obs[0].shape[0]

        i = self.obs_dim_per_agent

        res = torch.zeros([batch_size, self.numAgents*i], device=self.device)
        for a, obs_a in enumerate(obs):
            res[:,a*i:(a+1)*i] = obs_a
        return res
    
    def computeActions(self, learned_dynamics):
        na = self.numAgents
        batch_size = learned_dynamics.shape[0]
        
        # Replicate real PHS for batch
        F_sys_pinv = self.F_sys_pinv.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        R_sys = self.R_sys.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        J_sys = self.J_sys.unsqueeze(dim=0).repeat(batch_size, 1, 1)

        # Real dynamics
        dHdx_sys = torch.cat((torch.zeros(learned_dynamics.shape[0], int(learned_dynamics.shape[1]/2), device=self.device),
                                   learned_dynamics[:, :self.action_dim_per_agent * na]), dim=1).unsqueeze(2)
        real_dyn = torch.bmm(J_sys - R_sys, dHdx_sys)

        action_batch = torch.bmm(F_sys_pinv, learned_dynamics.unsqueeze(2) - real_dyn).reshape(batch_size, na, self.action_dim_per_agent)
        return action_batch
    
    def __build_real_dynamics(self):
        na = self.numAgents
        drag = 1 # TODO: Find appropiate value

        self.F_sys_pinv = torch.cat((torch.zeros(self.action_dim_per_agent * na,
                                                 self.action_dim_per_agent * na,
                                                 device=self.device),
                                 torch.eye(self.action_dim_per_agent * na, device=self.device)), dim=1)

        self.J_sys = torch.cat((torch.cat((torch.zeros(self.action_dim_per_agent * na,
                                                       self.action_dim_per_agent * na,
                                                       device=self.device),
                                 torch.eye(self.action_dim_per_agent * na, device=self.device)), dim=1),
                                torch.cat((-torch.eye(self.action_dim_per_agent * na, device=self.device),
                                torch.zeros(self.action_dim_per_agent * na,
                                            self.action_dim_per_agent * na, device=self.device)), dim=1)
                                ), dim=0)
        self.R_sys = torch.cat((torch.cat((torch.zeros(self.action_dim_per_agent * na,
                                                       self.action_dim_per_agent * na,
                                                       device=self.device),
                                 torch.zeros(self.action_dim_per_agent * na,
                                             self.action_dim_per_agent * na,
                                             device=self.device)), dim=1),
                                torch.cat((torch.zeros(self.action_dim_per_agent * na,
                                                       self.action_dim_per_agent * na,
                                                       device=self.device),
                                drag*torch.eye(self.action_dim_per_agent * na, device=self.device)), dim=1)
                                ), dim=0)

    # Environment management
    # ======================
    @abstractmethod 
    def setWorldStates(inputs):
        pass

    def setupEnvs(self, inputs):
        num_envs = inputs.shape[0]
        env_kwargs = self.getVMASConfig()

        self.env = make_env(
            scenario= self.vmas_scenario,
            num_envs=num_envs,
            device=self.device,
            continuous_actions=True,
            clamp_actions=True,
            grad_enabled=True,
            terminated_truncated=True,
            # Environment specific variables
            n_agents=self.numAgents,
            max_steps = self.episode_difficulty-1,
            **env_kwargs
        )

        for env_idx in range(self.env.num_envs):
            self.setWorldStates(self.env, env_idx, inputs[env_idx])
        return self.env
    
    # Returns all zeros. Implement your own in 
    # the inherited class to set 
    def randomInitialState(self, batch_size=1):
        i = self.agent_input_size
        na = self.numAgents
        obs = []
        for agent in range(na):
            agent_obs = torch.zeros(i)
            obs.append(agent_obs.unsqueeze(0))
        return tuple(obs)

    # def gif(self, trajectory):
    #         frames = []
    #         for i in range(trajectory.shape[0]):
    #             obs = trajectory[i,:].unsqueeze(0)
    #             #print(obs)
    #             env = self.setupEnvs(obs)
    #             frame = env.render(mode="rgb_array")
    #             frames.append(frame)
    #         return frames

    def gif(self, trajectory):
        frames = []
        
        # Determine the number of frames to render. Use the full trajectory length now.
        max_frames_to_render = trajectory.shape[0] 

        # Create the environment instance only once for the entire trajectory
        env_for_gif_rendering = None
        try:
            # Call setupEnvs ONLY ONCE here to get the environment
            initial_obs_for_env_setup = trajectory[0,:].unsqueeze(0)
            env_for_gif_rendering = self.setupEnvs(initial_obs_for_env_setup) 
            
            if not (hasattr(env_for_gif_rendering, 'render') and callable(getattr(env_for_gif_rendering, 'render'))):
                # This error indicates a fundamental problem, keep it.
                print(f"ERROR: Environment does not have a render method. Cannot generate GIFs.")
                return [] 

            for i in range(max_frames_to_render): 
                obs_for_current_frame = trajectory[i,:].unsqueeze(0)
                
                # Set the environment's state to match the current frame's observation
                # obs_for_current_frame[0] gets rid of the batch dimension (1, features) -> (features)
                self.setWorldStates(env_for_gif_rendering, 0, obs_for_current_frame[0]) 

                frame = env_for_gif_rendering.render(mode="rgb_array")
                
                # Keep these conversions for robustness, even if not strictly needed now
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                if frame.min() < 0 or frame.max() > 255:
                    frame = np.clip(frame, 0, 255)

                frames.append(frame)
            
            # Attempt to close the environment once after all frames for this trajectory
            if hasattr(env_for_gif_rendering, 'close') and callable(getattr(env_for_gif_rendering, 'close')):
                env_for_gif_rendering.close()
            # Removed the 'else' print as it's no longer a useful debug message
                
        except Exception as e:
            # Keep a general error message for unhandled exceptions during rendering
            print(f"ERROR: An exception occurred during rendering a trajectory: {e}")
            # Removed traceback.print_exc() for cleaner output in production
        finally:
            # Final safety check for closing, even if redundant
            if env_for_gif_rendering is not None and hasattr(env_for_gif_rendering, 'close') and callable(getattr(env_for_gif_rendering, 'close')):
                env_for_gif_rendering.close()

        return frames    
