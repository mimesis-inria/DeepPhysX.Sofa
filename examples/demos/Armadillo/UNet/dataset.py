"""
dataset.py
Run the data generation session to produce a Dataset only.
Use 'python3 dataset.py' to produce training Dataset (default).
Use 'python3 dataset.py -v' to produce validation Dataset.
"""

# Python related imports
import os
import sys

# DeepPhysX related imports
from DeepPhysX.Core.Pipelines.BaseDataGenerator import BaseDataGenerator
from DeepPhysX.Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX.Core.Visualizer.VedoVisualizer import VedoVisualizer
from DeepPhysX.Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig

# Session related imports
from download import ArmadilloDownloader
ArmadilloDownloader().get_session('run')
from Environment.ArmadilloTraining import ArmadilloTraining

# Dataset parameters
nb_batches = {'Training': 500, 'Validation': 50}
batch_size = {'Training': 32, 'Validation': 10}


def launch_data_generation(dataset_dir, dataset_mode):

    # Environment configuration
    environment_config = SofaEnvironmentConfig(environment_class=ArmadilloTraining,
                                               visualizer=VedoVisualizer,
                                               as_tcp_ip_client=True,
                                               number_of_thread=10)

    # Dataset configuration
    dataset_config = BaseDatasetConfig(dataset_dir=dataset_dir,
                                       partition_size=1,
                                       use_mode=dataset_mode)

    # Create DataGenerator
    data_generator = BaseDataGenerator(session_name='sessions/armadillo_data_user',
                                       environment_config=environment_config,
                                       dataset_config=dataset_config,
                                       nb_batches=nb_batches[dataset_mode],
                                       batch_size=batch_size[dataset_mode])

    # Launch the data generation session
    data_generator.execute()


if __name__ == '__main__':

    # Define dataset
    user_session = 'sessions/armadillo_data_user'
    dataset = user_session if os.path.exists(user_session) else None

    # Get dataset mode
    mode = 'Training'
    if len(sys.argv) > 1:
        if sys.argv[1] != '-v':
            print("Script option must be '-v' to produce validation dataset."
                  "By default, training dataset is produced.")
            quit(0)
        mode = 'Validation'

    # Launch pipeline
    launch_data_generation(dataset, mode)
