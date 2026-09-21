import copy
import json
import tempfile
from pathlib import Path
import unittest
import urllib.error

from jev_drive.api import JevClient, JevError, validate_choice
from jev_drive.evaluation import binary_metrics, partition_groups
from jev_drive.gate import Ticket, assess
from jev_drive.replay import load_records, run_replay
from jev_drive.protocol import CANDIDATES


def response():
    return {'model':'test-version', 'answers': {'decision': {'type':'choice', 'choice':'observe',
            'probabilities': {'continue':0.1,'replan':0.1,'observe':0.7,'defer':0.1},
            'confidence':0.8}}, 'usage': {'input_tokens':100}}


class ApiTests(unittest.TestCase):
    def test_candidate_identity_survives_response_reordering(self):
        result = validate_choice(response(), dict(reversed(list(CANDIDATES.items()))))
        self.assertEqual(result.choice, 'observe')
        self.assertEqual(result.model, 'test-version')

    def test_rejects_invalid_or_unoffered_answers(self):
        for field, value in [('choice','invented'),('confidence',float('nan')),
                             ('probabilities',{'continue':1}),('confidence',True),
                             ('probabilities',dict.fromkeys(CANDIDATES,0.8))]:
            with self.subTest(field=field, value=value):
                raw = response()
                raw['answers']['decision'][field] = value
                with self.assertRaises(JevError):
                    validate_choice(raw, CANDIDATES)

    def test_http_error_is_explicit_and_does_not_expose_key(self):
        def fail(payload):
            raise urllib.error.HTTPError('https://api.typesafe.ai',429,'SECRET',None,None)
        with self.assertRaisesRegex(JevError, '^http_429$'):
            JevClient(api_key='private-placeholder',transport=fail).choice({},'Question',CANDIDATES)

    def test_key_absent_from_payload_and_bad_timeouts_rejected(self):
        payloads = []
        client = JevClient(api_key='private-placeholder',transport=lambda p: payloads.append(p) or response())
        client.choice({'description':'road'},'Question',CANDIDATES)
        self.assertNotIn('private-placeholder', json.dumps(payloads))
        for timeout in [0,-1,float('nan'),float('inf'),True]:
            with self.assertRaises(ValueError):
                JevClient(api_key='placeholder',timeout=timeout)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.ticket = Ticket('obs-1',10.0,('continue','observe'))
        self.arguments = {'now':10.1,'current_observation_id':'obs-1',
                          'available':{'continue','observe'},'feasible':{'continue':True,'observe':True}}

    def test_accepts_only_current_locally_valid_request(self):
        result = assess(self.ticket,'continue',**self.arguments)
        self.assertTrue(result.accepted)
        self.assertEqual(result.action,'continue')

    def test_rejects_late_superseded_unavailable_and_vetoed(self):
        cases = [({'now':10.5},'expired_observation'),
                 ({'now':9.9},'expired_observation'),
                 ({'current_observation_id':'obs-2'},'superseded_observation'),
                 ({'available':{'observe'}},'unavailable_candidate'),
                 ({'feasible':{}},'local_veto_or_unknown'),
                 ({'feasible':{'continue':False}},'local_veto_or_unknown')]
        for update, reason in cases:
            with self.subTest(reason=reason):
                result = assess(self.ticket,'continue',**(self.arguments | update))
                self.assertFalse(result.accepted)
                self.assertIsNone(result.action)
                self.assertEqual(result.reason,reason)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{'sample_id':str(i),'group_id':f'route-{i//3}','image_sha256':f'hash-{i}'} for i in range(18)]

    def test_group_split_is_disjoint_reproducible_and_complete(self):
        split = partition_groups(self.rows,seed=17)
        self.assertEqual(split,partition_groups(self.rows,seed=17))
        groups = [{r['group_id'] for r in rows} for rows in split.values()]
        self.assertTrue(all(groups))
        self.assertTrue(groups[0].isdisjoint(groups[1]) and groups[0].isdisjoint(groups[2]) and groups[1].isdisjoint(groups[2]))
        self.assertEqual(sum(map(len,split.values())),len(self.rows))

    def test_duplicate_image_across_splits_is_rejected(self):
        for row in self.rows:
            row['image_sha256'] = 'same-image'
        with self.assertRaisesRegex(ValueError,'Identical image'):
            partition_groups(self.rows)

    def test_outcome_metrics_require_labels_and_match_known_values(self):
        result = binary_metrics([0.1,0.9],[0,1])
        self.assertAlmostEqual(result['brier'],0.01)
        self.assertAlmostEqual(result['ece'],0.1)
        with self.assertRaises(ValueError):
            binary_metrics([0.7],[])


class ReplayTests(unittest.TestCase):
    def test_outcome_fields_never_enter_inference_and_failures_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'input.jsonl'
            row = {'sample_id':'a','state':{'task':'drive','visual_description':'trail',
                                         'observation_scope':'single image'},'outcome':{'collision':True}}
            source.write_text(json.dumps(row),encoding='utf-8')
            captured = []
            def transport(payload):
                captured.append(payload)
                if len(captured) == 2:
                    raise OSError('sensitive internal data')
                return response()
            client = JevClient(api_key='placeholder',transport=transport)
            output = Path(directory)/'output.jsonl'
            summary = run_replay(source,output,client=client)
            self.assertEqual((summary['attempts'],summary['successes'],summary['errors']),(2,1,1))
            self.assertNotIn('collision',json.dumps(captured))
            self.assertNotIn('sensitive',output.read_text())
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertIsNone(records[1]['choice'])
            row['state']['future_label'] = 'turn left'
            source.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                load_records(source)


if __name__ == '__main__':
    unittest.main()
