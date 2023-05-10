import Sofa
import Sofa.Simulation

from DeepPhysX.Core.Environment.BaseEnvironment import BaseEnvironment


class SofaEnvironment(Sofa.Core.Controller, BaseEnvironment):

    def __init__(self, *args, **kwargs):
        """
        SofaEnvironment computes simulated data with SOFA for the Network and its training process.
        """

        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        # Warning: Define root node before init Environment
        self.root = Sofa.Core.Node('root')
        BaseEnvironment.__init__(self, **kwargs)
        self.root.addObject(self)

    def init(self) -> None:
        """
        Initialize the Environment. Automatically called when Environment is launched.
        """

        # Init the root node
        Sofa.Simulation.init(self.root)

    async def step(self):
        """
        Compute the number of steps in the Environment specified by simulations_per_step in EnvironmentConfig.
        """

        Sofa.Simulation.animate(self.root, self.root.dt.value)
        await self.on_step()

    async def on_step(self):
        """
        Executed after an animation step.
        No mandatory.
        """

        pass

    def update_visualisation(self) -> None:
        """
        Triggers the Visualizer update.
        """

        # The Sofa UserAPI does automatic updates
        pass
