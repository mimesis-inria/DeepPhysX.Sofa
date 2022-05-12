from typing import Optional, Any
from numpy import ndarray

from DeepPhysX_Core.Environment.BaseEnvironment import BaseEnvironment

import Sofa
import Sofa.Simulation


class SofaEnvironment(Sofa.Core.Controller, BaseEnvironment):
    """
    | SofaEnvironment is an environment class base on SOFA to compute simulated data for the network and its
      optimization process.

    :param Sofa.Core.Node root_node: Node used to create the scene graph
    :param str ip_address: IP address of the TcpIpObject
    :param int port: Port number of the TcpIpObject
    :param int instance_id: ID of the instance
    :param int number_of_instances: Number of simultaneously launched instances
    :param bool as_tcp_ip_client: Environment is owned by a TcpIpClient if True, by an EnvironmentManager if False
    :param Optional[Any] environment_manager: EnvironmentManager that handles the Environment if 'as_tcpip_client' is
                                              False
    """

    def __init__(self,
                 root_node: Sofa.Core.Node,
                 ip_address: str = 'localhost',
                 port: int = 10000,
                 instance_id: int = 0,
                 number_of_instances: int = 1,
                 as_tcp_ip_client: bool = True,
                 environment_manager: Optional[Any] = None,
                 *args, **kwargs):

        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        # Warning: Define root node before init Environment
        self.root = root_node
        BaseEnvironment.__init__(self,
                                 ip_address=ip_address,
                                 port=port,
                                 instance_id=instance_id,
                                 number_of_instances=number_of_instances,
                                 as_tcp_ip_client=as_tcp_ip_client,
                                 environment_manager=environment_manager)

    def init(self):
        """
        | Initialize the Environment.
        | Not mandatory.
        """

        Sofa.Simulation.init(self.root)

    async def step(self):
        """
        | Compute the number of steps in the Environment specified by simulations_per_step in EnvironmentConfig.
        """

        await self.animate()
        await self.on_step()

    async def animate(self):
        """
        | Trigger an Animation step.
        """

        Sofa.Simulation.animate(self.root, self.root.dt.value)

    async def on_step(self):
        """
        | Executed after an animation step.
        | No mandatory.
        """

        pass

    def __str__(self):
        """
        :return: String containing information about the BaseEnvironmentConfig object
        """

        description = BaseEnvironment.__str__(self)
        return description
