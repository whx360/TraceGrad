import unittest
import importlib.util


@unittest.skipIf(importlib.util.find_spec('torch') is None,'PyTorch independent reference unavailable')
class ReferenceTests(unittest.TestCase):
    def test_independent_forward_gradient_and_updates(self):
        from tracegrad.verification import verify_reference
        result=verify_reference()
        self.assertEqual(result['status'],'passed')
        self.assertEqual(len(result['cases']),4)


if __name__=='__main__': unittest.main()
