# `tnfr_lfs.cli.compare` module
Command helpers for the ``compare`` sub-command.

## Functions
- `register_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser], *, config: Mapping[str, Any]) -> None`
  - Register the ``compare`` sub-command.
- `handle(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str`
  - Execute the ``compare`` command returning the rendered payload.

## Attributes
- `SUPPORTED_AB_METRICS = tuple(sorted(SUPPORTED_LAP_METRICS))`

