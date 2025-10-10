# Plugin architecture

The telemetry toolkit exposes a lightweight plugin interface that allows
integrators to extend the natural-frequency and coherence workflows without
coupling to the higher level application stack.  Plugins are expected to derive
from [`TNFRPlugin`](../src/tnfr_lfs/plugins/base.py) and provide metadata that
identifies the component at runtime.

## Contract overview

`TNFRPlugin` is an abstract dataclass that combines immutable metadata with a
mutable analytics state:

* `identifier`, `display_name`, and `version` uniquely describe the plugin.
* `nu_f` exposes the latest natural frequency mapping keyed by subsystem node.
* `coherence_index` stores the most recent coherence indicator associated with
  the plugin.

Lifecycle helpers are available so implementers can respond to pipeline events:

* `reset_state()` clears the cached spectral data and invokes `on_reset()`.
* `apply_natural_frequency_snapshot(snapshot)` copies data from a
  [`NaturalFrequencySnapshot`](../src/tnfr_lfs/core/epi.py) and triggers
  `on_nu_f_updated(snapshot)`.
* `apply_coherence_index(value, series=None)` and `apply_coherence_series(series)`
  update the coherence state before calling `on_coherence_updated(...)`.

Only the hooks need to be overridden; the default implementations manage the
state containers and ensure coherent behaviour across plugins.

## Helper utilities

Two helper functions wire plugins into the existing spectral flow so that each
instance can operate autonomously:

* `resolve_plugin_nu_f(...)` (in `tnfr_lfs.core.epi`) wraps
  `resolve_nu_f_by_node`, applies the resulting snapshot to the plugin, and
  returns the snapshot for further processing.
* `plugin_coherence_operator(...)` (in `tnfr_lfs.core.operators`) runs the
  smoothing operator on a numeric series and propagates the resulting coherence
  series to the plugin.

When finer-grained control is required the `apply_plugin_nu_f_snapshot(...)`
helper can be used to push an existing snapshot into the plugin without
recomputing it.

## Metadata registry

Plugins declare their dependencies on TNFR operators through the
`tnfr_lfs.plugins.register_plugin_metadata` helper (or its
`tnfr_lfs.plugins.plugin_metadata` decorator variant).  Each plugin registers a
sequence of operator identifiers that correspond to the public names exported by
[`tnfr_lfs.core.operators`](../src/tnfr_lfs/core/operators.py).  The identifiers
are validated against the module's `__all__` contents so pipelines can rely on
the metadata at runtime.  The canonical list of identifiers can be retrieved via
`tnfr_lfs.plugins.available_operator_identifiers()`.

```python
from tnfr_lfs.plugins import TNFRPlugin, plugin_metadata


@plugin_metadata(operators=["reception_operator", "coherence_operator"])
class ReceptionAwarePlugin(TNFRPlugin):
    ...
```

`tnfr_lfs.plugins.get_plugin_operator_requirements(PluginClass)` returns the
declared operator names for a registered plugin.  Pipelines can iterate over the
registry using `tnfr_lfs.plugins.iter_plugin_operator_requirements()` to build
dependency matrices or to ensure that all required operators are available
before activating a plugin instance.

## Example

```python
from tnfr_lfs.plugins import TNFRPlugin
from tnfr_lfs.core.epi import resolve_plugin_nu_f
from tnfr_lfs.core.operators import plugin_coherence_operator


class SamplePlugin(TNFRPlugin):
    def on_nu_f_updated(self, snapshot):
        print(f"Updated ν_f: {snapshot.dominant_frequency:.2f} Hz")

    def on_coherence_updated(self, coherence_index, series=None):
        print(f"New coherence index: {coherence_index:.3f}")


plugin = SamplePlugin("demo", "Sample", "0.1.0")
snapshot = resolve_plugin_nu_f(plugin, telemetry_record)
smoothed = plugin_coherence_operator(plugin, coherence_series)
```

This pattern guarantees that the plugin always reflects the most recent
natural-frequency and coherence information while remaining decoupled from the
rest of the processing pipeline.
