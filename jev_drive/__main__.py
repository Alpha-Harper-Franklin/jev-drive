import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Jev + autonomous driving research tools')
    commands = parser.add_subparsers(dest='command', required=True)
    demo = commands.add_parser('demo', help='Synthetic contract test, not a driving benchmark')
    demo.add_argument('--scenario', choices=['clear_route','blocked_route','changed_route','low_visibility'], default='blocked_route')
    demo.add_argument('--provider', choices=['nominal','rules','jev'], default='rules')
    demo.add_argument('--seed', type=int, default=0)
    demo.add_argument('--output', required=True)
    replay = commands.add_parser('replay', help='Replay caption records through the live Jev API')
    replay.add_argument('--input', required=True)
    replay.add_argument('--output', required=True)
    replay.add_argument('--model', default='jev-1.13.0')
    replay.add_argument('--orders', type=int, choices=[1,2], default=2)
    args = parser.parse_args()
    if args.command == 'demo':
        from .episode import run_episode
        path = Path(args.output)
        if path.exists():
            parser.error('Output already exists; choose a new evidence file')
        episode = run_episode(args.scenario, args.seed, args.provider)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(episode.report(), stream, indent=2, allow_nan=False)
        print(json.dumps({'scope':'synthetic_contract_test', **episode.metrics()}))
    else:
        from .replay import run_replay
        summary = run_replay(args.input, args.output, args.model, args.orders)
        print(json.dumps(summary))
        if summary.get('errors'):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
