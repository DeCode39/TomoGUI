"""Check tag/package/changelog versions and optionally write release notes."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--notes', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = ast.parse((root / 'src/tomogui/__init__.py').read_text())
    version = next(ast.literal_eval(n.value) for n in source.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == '__version__' for t in n.targets))
    metadata = re.search(r'^version\s*=\s*"([^"]+)"', (root / 'pyproject.toml').read_text(), re.M).group(1)
    if args.tag != 'v' + version or metadata != version:
        raise SystemExit(f'Version mismatch: tag={args.tag}, module={version}, metadata={metadata}')
    changelog = (root / 'CHANGELOG.md').read_text()
    section = re.search(r'^## \[' + re.escape(version) + r'\][^\n]*\n(.*?)(?=^## |\Z)', changelog, re.M | re.S)
    if section is None:
        raise SystemExit('The changelog must contain a section for the release version.')
    if args.notes:
        args.notes.parent.mkdir(parents=True, exist_ok=True)
        args.notes.write_text(section[1].strip() + '\n', encoding='utf-8')
    print(f'Version checks passed: {args.tag}')


if __name__ == '__main__':
    main()
