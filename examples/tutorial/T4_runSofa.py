"""
#04 - Run Sofa Gui
Launch the Environment in a SOFA GUI.
The launched Environment should not perform requests during steps.
"""

# Python related imports
import os

# Sofa related imports
import Sofa.Gui

# DeepPhysX related imports
from DeepPhysX_Core.Manager.EnvironmentManager import EnvironmentManager

# Working session imports
from T3_configuration import env_config


def create_environment():
    # Environment should not be a TcpIpClient
    env_config.as_tcp_ip_client = False

    # Create DummyEnvironment within EnvironmentManager
    environment_manager = EnvironmentManager(environment_config=env_config)
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

    # Delete log files
    for file in os.listdir(os.getcwd()):
        if '.ini' in file or '.log' in file:
            os.remove(file)
