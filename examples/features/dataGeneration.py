"""
dataGeneration.py
Run the pipeline DataGenerator to produce a Dataset only.
"""

# DeepPhysX related imports
from DeepPhysX_Core.Pipelines.BaseDataGenerator import BaseDataGenerator
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Core.Visualizer.VedoVisualizer import VedoVisualizer
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig

# Session related imports
from Environment.EnvironmentDataset import MeanEnvironmentDataset


def launch_data_generation():
    # Define the number of points and the dimension
    nb_points = 30
    dimension = 3
    # Environment configuration
    environment_config = SofaEnvironmentConfig(environment_class=MeanEnvironmentDataset,
                                               visualizer=VedoVisualizer,
                                               param_dict={'constant': False,
                                                           'data_size': [nb_points, dimension],
                                                           'sleep': False},
                                               as_tcp_ip_client=True,
                                               number_of_thread=4)
    # Dataset configuration
    dataset_config = BaseDatasetConfig(normalize=False)
    # Create DataGenerator
    data_generator = BaseDataGenerator(session_name='sessions/data_generation',
                                       environment_config=environment_config,
                                       dataset_config=dataset_config,
                                       nb_batches=500,
                                       batch_size=10)
    # Launch the training session
    data_generator.execute()


if __name__ == '__main__':
    launch_data_generation()
