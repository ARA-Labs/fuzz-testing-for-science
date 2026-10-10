"""Run with python -m unittest discover -s tests; requires NumPy, not DiscoverPhysics."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import instrument


class GateResidualCoverage(unittest.TestCase):
    def messages(self):
        case = dict(probe_positions=[[1., 0.], [2., 0.]],
                    probe_velocities=[[0., 0.], [0., 0.]],
                    measurement_times=[1., 2.])
        out = dict(background_initial_positions=[[0., 0.]],
                   background_initial_velocities=[[0., 0.]],
                   positions=[[[0., 0.], [0., 0.], [0., 0.]],
                              [[0., 0.], [0., 0.], [0., 0.]]])
        return [dict(role="assistant", content="<run_experiment>" + json.dumps([case])
                     + "</run_experiment>"),
                dict(role="user", content="<experiment_output>" + json.dumps([out])
                     + "</experiment_output>")]

    def gate(self, law, messages=None):
        # Only replace the external candidate compiler, not NumPy residual math.
        with patch.object(instrument, "_compile", return_value=law):
            return instrument.gate_check(messages or self.messages(),
                                         "<final_law>fixture</final_law>", 1.)

    def test_matching_candidate_passes(self):
        self.assertEqual(self.gate(lambda **kwargs: [[0., 0.]] * 3)["action"], "pass")

    def test_late_probe_residual_is_not_discarded(self):
        def law(duration, **kwargs):
            return [[0., 0.], [100. if duration == 2. else 0., 0.], [0., 0.]]
        experiments, _ = instrument.parse_history(self.messages())
        self.assertEqual(instrument._predict_errors(law, *experiments[0]),
                         [0., 0., 10000., 0.])
        result = self.gate(law)
        self.assertEqual(result["action"], "bounce")
        self.assertEqual(result["worst_t"], 2.)
        self.assertGreater(result["max_sigma"], 20.)

    def test_second_probe_keeps_its_own_time(self):
        def law(duration, **kwargs):
            return [[0., 0.], [0., 0.], [100. if duration == 1. else 0., 0.]]
        result = self.gate(law)
        self.assertEqual(result["action"], "bounce")
        self.assertEqual(result["worst_t"], 1.)

    def test_nan_candidate_cannot_pass(self):
        self.assertIsNone(self.gate(lambda **kwargs: [[0., 0.], [float("nan"), 0.], [float("nan"), 0.]]))

    def test_infinite_residual_still_bounces(self):
        result = self.gate(lambda **kwargs: [[0., 0.], [float("inf"), 0.], [float("inf"), 0.]])
        self.assertEqual(result["action"], "bounce")

    def test_two_particle_protocol_is_unchanged(self):
        case = dict(pos2=[1., 0.], velocity2=[0., 0.], p1=1., p2=1.,
                    measurement_times=[1., 2.])
        out = dict(pos2=[[0., 0.], [0., 0.]])
        messages = [dict(role="assistant", content="<run_experiment>" + json.dumps([case])
                         + "</run_experiment>"),
                    dict(role="user", content="<experiment_output>" + json.dumps([out])
                         + "</experiment_output>")]
        def law(duration, **kwargs):
            return ([100. if duration == 2. else 0., 0.], [0., 0.])
        result = self.gate(law, messages)
        self.assertEqual(result["action"], "bounce")
        self.assertEqual(result["worst_t"], 2.)


if __name__ == "__main__":
    unittest.main()
