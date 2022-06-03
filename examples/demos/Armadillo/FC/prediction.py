"""
prediction.py
Launch the prediction session in a SOFA GUI with only predictions of the network.
Use 'python3 prediction.py' to render predictions in a SOFA GUI (default).
Use 'python3 validation.py -v' to render predictions with Vedo.
"""

# Python related imports
import os
import sys

# Sofa related imports
import Sofa.Gui

# DeepPhysX related imports
from DeepPhysX.Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX.Core.Visualizer.VedoVisualizer import VedoVisualizer
from DeepPhysX.Core.Pipelines.BaseRunner import BaseRunner
from DeepPhysX.Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig
from DeepPhysX.Sofa.Pipeline.SofaRunner import SofaRunner
from DeepPhysX.Torch.FC.FCConfig import FCConfig

# Session related imports
from download import ArmadilloDownloader
ArmadilloDownloader().get_session('run')
from Environment.ArmadilloPrediction import ArmadilloPrediction, np


def create_runner(visualizer=False):

    # Environment config
    env_config = SofaEnvironmentConfig(environment_class=ArmadilloPrediction,
                                       param_dict={'visualizer': visualizer},
                                       visualizer=VedoVisualizer if visualizer else None,
                                       as_tcp_ip_client=False)

    # Get the data size
    env_instance = env_config.create_environment(None)
    input_size, output_size = env_instance.input_size, env_instance.output_size
    env_instance.close()
    del env_instance

    # FC config
    nb_hidden_layers = 3
    nb_neurons = np.multiply(*input_size)
    nb_final_neurons = np.multiply(*output_size)
    layers_dim = [nb_neurons] + [nb_neurons for _ in range(nb_hidden_layers)] + [nb_final_neurons]
    net_config = FCConfig(network_name='armadillo_FC',
                          dim_output=3,
                          dim_layers=layers_dim,
                          biases=True)

    # Dataset config
    dataset_config = BaseDatasetConfig(normalize=True)

    # Define trained network session
    dpx_session = 'armadillo_dpx'
    user_session = 'armadillo_training_user'
    # Take user session by default
    session_name = user_session if os.path.exists('sessions/' + user_session) else dpx_session

    # Runner
    if visualizer:
        return BaseRunner(session_dir='sessions',
                          session_name=session_name,
                          dataset_config=dataset_config,
                          environment_config=env_config,
                          network_config=net_config,
                          nb_steps=100)
    else:
        return SofaRunner(session_dir='sessions',
                          session_name=session_name,
                          dataset_config=dataset_config,
                          environment_config=env_config,
                          network_config=net_config,
                          nb_steps=0)


if __name__ == '__main__':

    # Check missing data
    ArmadilloDownloader().get_session('predict')

    # Get option
    visualizer = False
    if len(sys.argv) > 1:
        # Check script option
        if sys.argv[1] != '-v':
            print("Script option must be '-v' to visualize predictions in a Vedo window."
                  "By default, prediction are rendered in a SOFA GUI.")
            quit(0)
        visualizer = True

    if visualizer:

        # Create and launch runner
        runner = create_runner(visualizer)
        runner.execute()
        runner.close()

    else:

        # Create SOFA runner
        runner = create_runner()

        # Launch SOFA GUI
        Sofa.Gui.GUIManager.Init("main", "qglviewer")
        Sofa.Gui.GUIManager.createGUI(runner.root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1080, 1080)
        Sofa.Gui.GUIManager.MainLoop(runner.root)
        Sofa.Gui.GUIManager.closeGUI()

        # Manually close the runner (security if stuff like additional dataset need to be saved)
        runner.close()

        # Delete unwanted files
        for file in os.listdir(os.path.dirname(os.path.abspath(__file__))):
            if '.ini' in file or '.log' in file:
                os.remove(file)
