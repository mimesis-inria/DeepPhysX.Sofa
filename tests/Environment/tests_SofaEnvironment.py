from unittest import TestCase
from asyncio import run
import numpy as np
import Sofa

from DeepPhysX_Sofa.Environment.SofaEnvironment import SofaEnvironment
from TestEnvironment import TestEnvironment


class TestSofaEnvironment(TestCase):

    def setUp(self):
        node = Sofa.Core.Node()
        self.env = node.addObject(TestEnvironment(root_node=node, as_tcp_ip_client=False))

    def test_init(self):
        # Default values at init
        for attribute in [self.env.input, self.env.output]:
            self.assertTrue(type(attribute), np.ndarray)
        for attribute in [self.env.sample_in, self.env.sample_out]:
            self.assertEqual(attribute, None)
        for attribute in [self.env.loss_data, self.env.environment_manager]:
            self.assertEqual(attribute, None)
        for attribute in [self.env.additional_fields]:
            self.assertEqual(attribute, {})
        self.assertIsInstance(self.env.root, Sofa.Core.Node)

    def test_not_implemented(self):
        # Check not implemented functions
        env = SofaEnvironment(root_node=Sofa.Core.Node(), as_tcp_ip_client=False)
        with self.assertRaises(NotImplementedError):
            env.create()

    def test_triggers(self):
        # Trigger main simulation methods
        self.env.create()
        self.env.init()
        run(self.env.step())
        for attribute in [self.env.call_create, self.env.call_init, self.env.call_step]:
            self.assertTrue(attribute)
