import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
MODELS = {
    'qwen': ('mlx-community/Qwen2.5-7B-Instruct-4bit', 'c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed'),
    'distilbert': ('distilbert/distilbert-base-uncased', '12040accade4e8a0f71eabdb258fecc2e7e948be'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=MODELS)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--output', type=Path, default=PACKAGE / 'outputs')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ['M17_OUTPUT'] = str(output)
    from common import GROUPS

    model_dir = args.model_dir
    if model_dir is None:
        from huggingface_hub import snapshot_download
        repo, revision = MODELS[args.model]
        model_dir = snapshot_download(
            repo, revision=revision,
            allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'],
        )
    model_dir = Path(model_dir).resolve()
    os.environ['M17_QWEN' if args.model == 'qwen' else 'M17_ENCODER'] = str(model_dir)
    manifest = output / f'{args.model}_run.json'
    if manifest.exists():
        parser.error('Output contains an existing run; choose a fresh output directory')
    manifest.write_text(json.dumps({
        'model': MODELS[args.model][0], 'revision': MODELS[args.model][1],
        'local_model_override': args.model_dir is not None, 'seeds': [544, 2026],
    }, indent=2) + '\n')

    def run(script, *arguments):
        name = '_'.join([Path(script).stem, *arguments]) if '--adapter' not in arguments else 'qwen_verify'
        print(f'Running {script} {" ".join(arguments)}', flush=True)
        with (output / f'{name}.log').open('w') as log:
            subprocess.run([sys.executable, str(PACKAGE / script), *arguments],
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    if args.model == 'distilbert':
        run('distilbert.py')
    else:
        run('qwen.py', 'baseline')
        run('qwen.py', 'reason')
        for seed in [544, 2026]:
            for group in GROUPS:
                run('qwen.py', 'train', '--group', group, '--seed', str(seed))
        adapter = output / f'qwen_{GROUPS[-1]}_2026'
        run('qwen.py', 'verify', '--adapter', str(adapter))
    subprocess.run([sys.executable, str(PACKAGE / 'summarize.py')], check=True)


if __name__ == '__main__':
    main()
