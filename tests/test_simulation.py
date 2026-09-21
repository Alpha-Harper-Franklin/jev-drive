import unittest
from jev_drive.episode import run_episode


class SyntheticContractTests(unittest.TestCase):
    def test_rules_execute_real_replanning_on_blocked_route(self):
        episode = run_episode('blocked_route',0,'rules')
        self.assertGreater(episode.replans,0)
        self.assertTrue(any(d.get('execution') == 'local_path_updated' for d in episode.decisions))
        self.assertEqual(episode.metrics()['api_attempts'],0)
        self.assertEqual(episode.status,'success')

    def test_sensor_outage_appears_in_decisions(self):
        episode = run_episode('low_visibility',0,'rules')
        self.assertTrue(any(d['action'] == 'observe' for d in episode.decisions))
        self.assertIn('range-limited exact geometry',episode.decisions[0]['observation']['provenance'])


if __name__ == '__main__':
    unittest.main()
