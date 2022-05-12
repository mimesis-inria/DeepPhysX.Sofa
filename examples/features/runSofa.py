"""
runSofa.py
Launch the Environment in a SOFA GUI.
The launched Environment should not perform requests during steps.
"""

# Python related imports
import os

# Sofa related imports
import Sofa.Gui

# DeepPhysX related imports
from DeepPhysX_Core.Manager.EnvironmentManager import EnvironmentManager
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig

# Working session imports
from Environment.EnvironmentSofa import MeanEnvironmentSofa


def create_environment():
    # Define the number of points and the dimension
    data_size = [30, 3]
    # Create SofaEnvironment configuration
    environment_config = SofaEnvironmentConfig(environment_class=MeanEnvironmentSofa,
                                               as_tcp_ip_client=False,
                                               param_dict={'constant': False,
                                                           'data_size': data_size,
                                                           'sleep': False})

    # Create Armadillo Environment within EnvironmentManager
    environment_manager = EnvironmentManager(environment_config=environment_config)
    return environment_manager.environment


if __name__ == '__main__':

    # Create Environment
    environment = create_environment()

    # Launch Sofa GUI
    Sofa.Gui.GUIManager.Init(program_name="main", gui_name="qglviewer")
    Sofa.Gui.GUIManager.createGUI(environment.root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    Sofa.Gui.GUIManager.MainLoop(environment.root)
    Sofa.Gui.GUIManager.closeGUI()

    # Close environment
    environment.close()

    # Delete log files
    for file in os.listdir(os.getcwd()):
        if '.ini' in file or '.log' in file:
            os.remove(file)
