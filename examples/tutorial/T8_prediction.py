"""
#08 - Prediction
Launch a running session in a SOFA GUI.
"""

# Python related imports
import os

# Sofa related imports
import Sofa.Gui

# DeepPhysX related imports
from DeepPhysX_Sofa.Pipeline.SofaRunner import SofaRunner

# Session related imports
from T3_configuration import env_config, net_config


def create_runner():
    # Environment should not be a TcpIpClient
    env_config.as_tcp_ip_client = False

    # Runner
    return SofaRunner(session_dir='sessions',
                      session_name='online_training',
                      environment_config=env_config,
                      network_config=net_config,
                      nb_steps=0)


if __name__ == '__main__':

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
