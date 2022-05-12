from typing import Type, Optional, Any, Dict
from os.path import join, dirname
from sys import modules, executable
from subprocess import call as subprocesscall

import Sofa

from DeepPhysX_Core.Environment.BaseEnvironmentConfig import BaseEnvironmentConfig
from DeepPhysX_Core.Visualizer.VedoVisualizer import VedoVisualizer
from DeepPhysX_Sofa.Environment.SofaEnvironment import SofaEnvironment


class SofaEnvironmentConfig(BaseEnvironmentConfig):
    """
    | SofaEnvironmentConfig is a configuration class to parameterize and create a SofaEnvironment for the
      EnvironmentManager.

    :param Type[SofaEnvironment] environment_class: Class from which an instance will be created
    :param Optional[Type[VedoVisualizer]] visualizer: Class of the Visualizer to use
    :param int simulations_per_step: Number of iterations to compute in the Environment at each time step
    :param int max_wrong_samples_per_step: Maximum number of wrong samples to produce in a step
    :param bool always_create_data: If True, data will always be created from environment. If False, data will be
                                    created from the environment during the first epoch and then re-used from the
                                    Dataset.
    :param bool record_wrong_samples: If True, wrong samples are recorded through Visualizer
    :param int screenshot_sample_rate: A screenshot of the viewer will be done every x sample
    :param bool use_dataset_in_environment: If True, the dataset will always be used in the environment
    :param Optional[Dict[Any, Any]] param_dict: Dictionary containing specific environment parameters
    :param bool as_tcp_ip_client: Environment is owned by a TcpIpClient if True, by an EnvironmentManager if False
    :param int number_of_thread: Number of thread to run
    :param int max_client_connection: Maximum number of handled instances
    :param str environment_file: Path of the file containing the Environment class
    :param str ip_address: IP address of the TcpIpObject
    :param int port: Port number of the TcpIpObject
    """

    def __init__(self,
                 environment_class: Type[SofaEnvironment],
                 visualizer: Optional[Type[VedoVisualizer]] = None,
                 simulations_per_step: int = 1,
                 max_wrong_samples_per_step: int = 10,
                 always_create_data: bool = False,
                 record_wrong_samples: bool = False,
                 screenshot_sample_rate: int = 0,
                 use_dataset_in_environment: bool = False,
                 param_dict: Optional[Dict[Any, Any]] = None,
                 as_tcp_ip_client: bool = True,
                 max_client_connection: int = 1000,
                 number_of_thread: int = 1,
                 environment_file: Optional[str] = None,
                 ip_address: str = 'localhost',
                 port: int = 10000):

        BaseEnvironmentConfig.__init__(self,
                                       environment_class=environment_class,
                                       visualizer=visualizer,
                                       simulations_per_step=simulations_per_step,
                                       max_wrong_samples_per_step=max_wrong_samples_per_step,
                                       always_create_data=always_create_data,
                                       record_wrong_samples=record_wrong_samples,
                                       screenshot_sample_rate=screenshot_sample_rate,
                                       use_dataset_in_environment=use_dataset_in_environment,
                                       param_dict=param_dict,
                                       as_tcp_ip_client=as_tcp_ip_client,
                                       number_of_thread=number_of_thread,
                                       max_client_connection=max_client_connection,
                                       environment_file=environment_file,
                                       ip_address=ip_address,
                                       port=port)

        self.environment_class: Type[SofaEnvironment] = environment_class

    def start_client(self, idx: int = 1) -> None:
        """
        | Run a subprocess to start a TcpIpClient.

        :param int idx: Index of client
        """

        script = join(dirname(modules[SofaEnvironment.__module__].__file__), 'launcherSofaEnvironment.py')
        # Usage: python3 script.py <file_path> <environment_class> <ip_address> <port> <idx> <nb_threads>"
        subprocesscall([executable,
                        script,
                        self.environment_file,
                        self.environment_class.__name__,
                        self.ip_address,
                        str(self.port),
                        str(idx),
                        str(self.number_of_thread)])

    def create_environment(self, environment_manager: Any) -> SofaEnvironment:
        """
        | Create an Environment that will not be a TcpIpObject.

        :param Any environment_manager: EnvironmentManager that handles the Environment
        :return: Environment object
        """

        # Create instance
        root_node = Sofa.Core.Node()
        try:
            environment = root_node.addObject(self.environment_class(environment_manager=environment_manager,
                                                                     root_node=root_node, as_tcp_ip_client=False))
        except:
            raise ValueError(f"[{self.name}] Given 'environment_class' cannot be created in {self.name}")
        if not isinstance(environment, SofaEnvironment):
            raise TypeError(f"[{self.name}] Wrong 'environment_class' type: SofaEnvironment required, get "
                            f"{self.environment_class}")
        # Create & Init Environment
        environment.recv_parameters(self.param_dict)
        environment.create()
        environment.init()
        return environment
