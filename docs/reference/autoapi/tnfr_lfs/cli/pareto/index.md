# `tnfr_lfs.cli.pareto` module
Command helpers for the ``pareto`` sub-command.

## Functions
- `register_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser], *, config: Mapping[str, Any]) -> None`
  - Register the ``pareto`` sub-command.
- `handle(namespace: argparse.Namespace, *, config: Mapping[str, Any]) -> str`
  - Execute the ``pareto`` command returning the rendered payload.

