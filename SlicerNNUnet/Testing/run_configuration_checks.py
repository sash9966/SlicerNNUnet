"""Run synthetic Slicer checks without installing packages, weights or sample data.

Slicer --testing --no-splash --no-main-window --python-script /absolute/path/to/this/file
"""
import sys
import unittest
from pathlib import Path

import slicer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Testing.InstallLogicTestCase import InstallLogicTestCase
from Testing.ProcessTestCase import ProcessTestCase
from Testing.SegmentationLogicTestCase import SyntheticGeometryTestCase
from Testing.WidgetTestCase import WidgetTestCase

suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case) for case in (
    InstallLogicTestCase, ProcessTestCase, SyntheticGeometryTestCase, WidgetTestCase
))
result = unittest.TextTestRunner(verbosity=2).run(suite)
slicer.app.exit(0 if result.wasSuccessful() else 1)
