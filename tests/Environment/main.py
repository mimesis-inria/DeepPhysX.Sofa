import unittest
from os import devnull
from sys import stdout

from tests_SofaEnvironmentConfig import TestSofaEnvironmentConfig
from tests_SofaEnvironment import TestSofaEnvironment


if __name__ == '__main__':
    stdout = open(devnull, 'w')
    unittest.main()
