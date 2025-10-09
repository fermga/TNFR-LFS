# CLI deep dive

The README links to this document for the extended CLI workflows and
configuration reference.

## Configuration template

The repository ships `tnfr_lfs.toml` as a ready-to-use template mirroring the
options described in the [CLI guide](cli.md). Copy it to
`~/.config/tnfr_lfs.toml` or to your project root to customise OutSim/OutGauge/
InSim ports, initial car/track presets and ΔNFR limits per phase. Packs can also
be resolved at runtime by pointing `paths.pack_root` to a directory that contains
`config/global.toml`, `data/cars` and `data/profiles`—the CLI then embeds the
selected pack metadata in every JSON export.

## Pareto and compare subcommands

| Subcommand | Purpose | Recommended exporters | Example |
| --- | --- | --- | --- |
| `pareto` | Explore the decision space around a generated plan and export the Pareto front. | `markdown` for quick inspection or `html_ext` for interactive reports. | `tnfr_lfs pareto stint.jsonl --car-model FZR --radius 2 --export html_ext > pareto.html` |
| `compare` | Aggregate lap metrics from two stints/configurations and highlight the winning variant. | `markdown` for engineering notes or `html_ext` for dashboards. | `tnfr_lfs compare baseline.jsonl variant.jsonl --metric sense_index --export html_ext > abtest.html` |

Both commands support `--export html_ext`, bundling tables, histograms and
playbook suggestions into a single portable HTML document.

## Live HUD quickstart

Run the ΔNFR HUD inside Live for Speed by enabling the simulator broadcasters
and executing:

```bash
tnfr_lfs osd --host 127.0.0.1 --outsim-port 4123 --outgauge-port 3000 --insim-port 29999
```

The overlay cycles through phase status, nodal contributions and setup plan
pages that fit inside a 40×16 `IS_BTN` button. Move it with
`--layout-left`/`--layout-top` when other mods occupy the same area. For the full
HUD breakdown, consult [`docs/cli.md`](cli.md).

## Related documentation

* [CLI guide](cli.md)
* [Setup equivalences](setup_equivalences.md)
* [Examples gallery](examples_gallery.md)
