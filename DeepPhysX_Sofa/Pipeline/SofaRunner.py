from typing import Optional
import Sofa

from DeepPhysX_Core.Pipelines.BaseRunner import BaseRunner
from DeepPhysX_Core.Network.BaseNetworkConfig import BaseNetworkConfig
from DeepPhysX_Core.Dataset.BaseDatasetConfig import BaseDatasetConfig
from DeepPhysX_Sofa.Environment.SofaEnvironmentConfig import SofaEnvironmentConfig


class SofaRunner(Sofa.Core.Controller, BaseRunner):
    """
    | BaseRunner is a pipeline defining the running process of an artificial neural network.
    | It provides a highly tunable learning process that can be used with any machine learning library.

    :param BaseNetworkConfig network_config: Specialisation containing the parameters of the network manager
    :param environment_config: Specialisation containing the parameters of the environment manager
    :param Optional[BaseDatasetConfig] dataset_config: Specialisation containing the parameters of the dataset manager
    :param str session_name: Name of the newly created directory if session_dir is not defined
    :param Optional[str] session_dir: Name of the directory in which to write all the necessary data
    :param int nb_steps: Number of simulation step to play
    :param bool record_inputs: Save or not the input in a numpy file
    :param bool record_outputs: Save or not the output in a numpy file
    """

    def __init__(self,
                 network_config: BaseNetworkConfig,
                 environment_config: SofaEnvironmentConfig,
                 dataset_config: Optional[BaseDatasetConfig] = None,
                 session_name: str = 'default',
                 session_dir: Optional[str] = None,
                 nb_steps: int = 0,
                 record_inputs: bool = False,
                 record_outputs: bool = False,
                 *args, **kwargs):

        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        BaseRunner.__init__(self,
                            network_config=network_config,
                            dataset_config=dataset_config,
                            environment_config=environment_config,
                            session_name=session_name,
                            session_dir=session_dir,
                            nb_steps=nb_steps,
                            record_inputs=record_inputs,
                            record_outputs=record_outputs)
        self.run_begin()
        self.root = self.manager.data_manager.environment_manager.environment.root
        self.root.addObject(self)

    def onAnimateEndEvent(self, event):
        """
        | Called within the Sofa pipeline at the end of the time step.

        :param event: Sofa Event
        """

        if self.running_condition():
            self.sample_begin()
            prediction = self.predict(animate=False)
            self.sample_end(prediction)
        else:
            self.run_end()
