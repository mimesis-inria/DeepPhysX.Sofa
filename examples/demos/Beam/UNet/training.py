"""
training.py
Launch the training session with a VedoVisualizer.
Use 'python3 training.py' to run the pipeline with existing samples from a Dataset (default).
Use 'python3 training.py <nb_thread>' to run the pipeline with newly created samples in Environment.
"""

# Python related imports
import os.path
import sys
import torch

# DeepPhysX related imports
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Visualizer.VedoVisualizer import VedoVisualizer
from DeepPhysX_Core.Pipelines.BaseTrainer import BaseTrainer
from DeepPhysX_Torch.UNet.UNetConfig import UNetConfig
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig

# Working session imports
from download import BeamDownloader
from Environment.BeamTraining import BeamTraining
from Environment.parameters import grid_resolution

# Training parameters
nb_epochs = 400
nb_batch = 2000
batch_size = 8
lr = 1e-5


def launch_trainer(dataset_dir, nb_env):

    # Environment config
    env_config = SofaEnvironmentConfig(environment_class=BeamTraining,
                                       visualizer=VedoVisualizer,
                                       number_of_thread=nb_env)

    # UNet config
    net_config = UNetConfig(network_name='beam_UNet',
                            loss=torch.nn.MSELoss,
                            lr=lr,
                            optimizer=torch.optim.Adam,
                            input_size=grid_resolution.tolist(),
                            nb_dims=3,
                            nb_input_channels=3,
                            nb_first_layer_channels=128,
                            nb_output_channels=3,
                            nb_steps=3,
                            two_sublayers=True,
                            border_mode='same',
                            skip_merge=False)

    # Dataset config
    dataset_config = BaseDatasetConfig(partition_size=1,
                                       shuffle_dataset=True,
                                       normalize=True,
                                       dataset_dir=dataset_dir)

    # Trainer
    trainer = BaseTrainer(session_dir='sessions',
                          session_name='beam_training_user',
                          dataset_config=dataset_config,
                          environment_config=env_config,
                          network_config=net_config,
                          nb_epochs=nb_epochs,
                          nb_batches=nb_batch,
                          batch_size=batch_size)

    # Launch the training session
    trainer.execute()


if __name__ == '__main__':

    # Define dataset
    dpx_session = 'sessions/beam_dpx'
    user_session = 'sessions/beam_data_user'
    # Take user dataset by default
    dataset = user_session if os.path.exists(user_session) else dpx_session

    # Get nb_thread options
    nb_thread = 1
    if len(sys.argv) > 1:
        dataset = None
        try:
            nb_thread = int(sys.argv[1])
        except ValueError:
            print("Script option must be an integer <nb_sample> for samples produced in Environment(s)."
                  "Without option, samples are loaded from an existing Dataset.")
            quit(0)

    # Check missing data
    session_name = 'train' if dataset is None else 'train_data'
    BeamDownloader().get_session(session_name)

    # Launch pipeline
    launch_trainer(dataset, nb_thread)
