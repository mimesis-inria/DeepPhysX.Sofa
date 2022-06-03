"""
training.py
Launch the training session with a VedoVisualizer.
Use 'python3 training.py <nb_thread>' to run the pipeline with newly created samples in Environment (default).
Use 'python3 training.py -d' to run the pipeline with existing samples from a Dataset.
"""

# Python related imports
import os.path
import sys
import torch

# DeepPhysX related imports
from DeepPhysX.Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX.Core.Visualizer.VedoVisualizer import VedoVisualizer
from DeepPhysX.Core.Pipelines.BaseTrainer import BaseTrainer
from DeepPhysX.Torch.FC.FCConfig import FCConfig
from DeepPhysX.Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig

# Working session imports
from download import BeamDownloader
from Environment.BeamTraining import BeamTraining, p_grid

# Training parameters
nb_epochs = 400
nb_batch = 500
batch_size = 32
lr = 1e-5


def launch_trainer(dataset_dir, nb_env):

    # Environment config
    env_config = SofaEnvironmentConfig(environment_class=BeamTraining,
                                       visualizer=VedoVisualizer,
                                       number_of_thread=nb_env)

    # FC config
    nb_hidden_layers = 2
    nb_neurons = p_grid.nb_nodes * 3
    layers_dim = [nb_neurons] + [nb_neurons for _ in range(nb_hidden_layers + 1)] + [nb_neurons]
    net_config = FCConfig(network_name='beam_FC',
                          loss=torch.nn.MSELoss,
                          lr=lr,
                          optimizer=torch.optim.Adam,
                          dim_output=3,
                          dim_layers=layers_dim,
                          biases=True)

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
