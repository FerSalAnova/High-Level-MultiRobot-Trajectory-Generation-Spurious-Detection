import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

from abc import ABC, abstractmethod
import torch, imageio

from Training.PathManager import PathManager
from Task.TaskBuilder import TaskBuilder
import numpy as np
import traceback

class Generator(ABC):
    def __init__(self, config):
        self.config = config
                
        # Path manager
        self.path_manager = PathManager(self.config)
        
        self.na = int(self.config["task"]["num_agents"])
        self.episode_difficulty = int(self.config["task"]["episode_difficulty"])
        
        self.numTrain = int(self.config["general"]["train_size"])
        self.numVal = int(self.config["general"]["val_size"])
        self.numTest = int(self.config["general"]["test_size"])
        
        self.task = TaskBuilder(config)

        self.seed = int(self.config["general"]["seed_data"])
        torch.manual_seed(self.seed)

        self.device = torch.device(config["general"]["device"] if torch.cuda.is_available() else 'cpu')

        try:
            os.makedirs(self.path_manager.getPathDatasets())
        except FileExistsError:
            pass 

    @abstractmethod
    def generateDataset(self, numData):
        pass
    

    def generateTrainValTest(self):
        if self.task.robot_obs_noise != 0 or self.task.action_noise_factor:
            self.generateNoisyTrainValTest()
            return
        
        # Generate noiseless dataset
        train_obs, train_actions, _ = self.generateDataset(self.numTrain)
        torch.save(train_obs, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("train", noisy=False))
        torch.save(train_actions, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("train", data="actions", noisy=False))
        torch.save(train_obs, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("train", noisy=True))
        torch.save(train_actions, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("train", data="actions", noisy=True))
        self.saveGifs(train_obs)


        val_obs, val_actions, _ = self.generateDataset(self.numVal)
        torch.save(val_obs, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("val", noisy=False))
        torch.save(val_actions, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("val", data="actions", noisy=False))
        torch.save(val_obs, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("val", noisy=True))
        torch.save(val_actions, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("val", data="actions", noisy=True))
        #self.saveGifs(val_obs)

        test_obs, test_actions, test_rewards = self.generateDataset(self.numTest)
        torch.save(test_obs, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("test", noisy=False))
        torch.save(test_actions, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("test", data="actions", noisy=False))
        torch.save(test_rewards, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("test", data="rewards", noisy=False))
        torch.save(test_obs, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("test", noisy=True))
        torch.save(test_actions, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("test", data="actions", noisy=True))
        #self.saveGifs(test_obs)

    def generateValidation(self):
        val_data = self.generateDataset(self.numVal)
        torch.save(val_data, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("val"))
        torch.save(val_data, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("val", False))

    # Noisy observations
    def generateNoisyTrainValTest(self):
        # Temporally update config to access or generate data without noise
        print("Generating noisy dataset, looking for an existing one to denoise...")
        old_noise_factor = self.task.robot_obs_noise
        old_action_noise_factor = self.task.action_noise_factor
        self.task.robot_obs_noise = 0
        self.task.action_noise_factor = 0
        self.config["task"]["robot_obs_noise"] = 0
        self.config["task"]["action_noise_factor"] = 0
        
        if not os.path.exists(self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("train")):
            print("Datasets without noise not found. Generating them...")
            self.generateTrainValTest()

        train_obs_noNoise = torch.load(self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("train", noisy=False).to(self.device))
        train_act_noNoise = torch.load(self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("train", data="actions", noisy=False)).to(self.device)
        val_obs_noNoise = torch.load(self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("val", noisy=False)).to(self.device)
        val_act_noNoise = torch.load(self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("val", data="actions", noisy=False)).to(self.device)
        test_obs_noNoise = torch.load(self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("test", noisy=False)).to(self.device)
        test_act_noNoise = torch.load(self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("test", data="actions", noisy=False)).to(self.device)
        test_rew_noNoise = torch.load(self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("test", data="rewards", noisy=False)).to(self.device)
        print("Datasets without noise lodaded. Adding noise to observations...")



        # Return to original config and add noise
        self.task.robot_obs_noise = old_noise_factor
        self.config["task"]["robot_obs_noise"] =old_noise_factor
        self.task.action_noise_factor = old_action_noise_factor
        self.config["task"]["action_noise_factor"] =old_action_noise_factor

        # Copy of noiseless data to new config (for evaluation with GT)
        torch.save(train_obs_noNoise, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("train",noisy=False))
        torch.save(train_act_noNoise, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("train",data="actions", noisy=False))
        torch.save(val_obs_noNoise, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("val", noisy=False))
        torch.save(val_act_noNoise, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("val",data="actions", noisy=False))
        torch.save(test_obs_noNoise, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("test", noisy=False))
        torch.save(test_act_noNoise, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("test",data="actions", noisy=False))
        torch.save(test_rew_noNoise, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("test",data="rewards", noisy=False))

        # Noisy datasets
        train_obs = self.task.addNoise(train_obs_noNoise)
        train_act = self.task.addNoise(train_act_noNoise, True)
        torch.save(train_obs, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("train", noisy=True))
        torch.save(train_act, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("train", data="actions", noisy=True))
        self.saveGifs(train_obs)
        
        val_obs =  self.task.addNoise(val_obs_noNoise)
        val_act = self.task.addNoise(val_act_noNoise, True)
        torch.save(val_obs, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("val", noisy=True))
        torch.save(val_act, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("val", data="actions", noisy=True))
        
        test_obs = self.task.addNoise(test_obs_noNoise)
        test_act = self.task.addNoise(test_act_noNoise, True)
        torch.save(test_obs, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("test", noisy=True))
        torch.save(test_act, self.path_manager.getPathDatasets()+self.path_manager.getDatasetFilename("test", data="actions", noisy=True))


    # def saveGifs(self, trajectories):
    #     task = TaskBuilder(self.config)
    #     if hasattr(task, "gif") and callable(getattr(task, "gif")):
    #         print("Generating gifs...")
    #         folder = self.path_manager.getPathDatasets()+"qualitative/"
    #         os.makedirs(folder, exist_ok=True)

    #         for i in range(5):
    #             trajetory = trajectories[:,i,:]
    #             frames = task.gif(trajetory)
    #             imageio.mimsave(folder+str(i)+".gif", frames, duration=15)

    def saveGifs(self, trajectories):
        task = TaskBuilder(self.config) # Assuming TaskBuilder is accessible
        if hasattr(task, "gif") and callable(getattr(task, "gif")):
            print("Generating gifs...") # Keep this as a general progress indicator
            base_dataset_path = self.path_manager.getPathDatasets()
            folder = os.path.join(base_dataset_path, "qualitative") 

            try:
                os.makedirs(folder, exist_ok=True)
            except OSError as e:
                print(f"Error creating directory {folder}: {e}")
                return 

            num_trajectories_to_process = min(5, trajectories.shape[1]) 

            for i in range(num_trajectories_to_process): 
                trajectory_single = trajectories[:, i, :] 
                
                try:
                    frames = task.gif(trajectory_single)
                    
                    if not frames:
                        # Keep this warning as it indicates a potential issue with frame generation
                        print(f"Warning: task.gif returned an EMPTY list of frames for trajectory {i}. Skipping GIF for this trajectory.")
                        continue 
                    
                    file_name = f"{i}.gif"
                    file_path = os.path.join(folder, file_name) 
                    
                    imageio.mimsave(file_path, frames, fps=15) 
                    print(f"Successfully saved GIF to: {file_path}") # Keep this confirmation
                except Exception as e:
                    # Keep a general error message for unhandled exceptions during saving
                    print(f"Error saving GIF for trajectory {i}: {e}")
        else:
            print("Task does not have a 'gif' method or it's not callable. Skipping GIF generation.")
