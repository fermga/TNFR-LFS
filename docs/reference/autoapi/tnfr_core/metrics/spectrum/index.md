# `tnfr_core.metrics.spectrum` module
Spectral helpers shared across modal and phase alignment analyses.

## Classes
### `PhaseCorrelation`
Correlation payload describing the dominant cross-spectrum component.

## Functions
- `phase_to_latency_ms(frequency: float, phase: float) -> float`
  - Convert a phase lag expressed in radians to milliseconds.
- `motor_input_correlations(records: Sequence[SupportsTelemetrySample], *, controls: Mapping[str, Sequence[str]] | None = None, responses: Mapping[str, Sequence[str]] | None = None, sample_rate: float | None = None, dominant_strategy: Literal['auto', 'fft', 'goertzel'] = 'auto', candidate_frequencies: Sequence[float] | None = None) -> Dict[Tuple[str, str], PhaseCorrelation]`
  - Return dominant correlations between driver inputs and chassis responses.

``dominant_strategy`` selects whether to evaluate the dense FFT
cross-spectrum (``"fft"``) or the sparse Goertzel helper when SciPy is
installed (``"auto"``/``"goertzel"``). ``candidate_frequencies`` constrains
the Goertzel search to custom bins when provided.
- `estimate_sample_rate(records: Sequence[SupportsTelemetrySample]) -> float`
  - Estimate the sampling frequency of ``records`` in Hertz.
- `hann_window(length: int, *, xp_module: Any | None = None) -> Any`
  - Return a Hann window matching ``length`` samples, evaluating with ``xp_module`` when provided.
- `apply_window(samples: Sequence[float], window: Sequence[float], *, xp_module: Any | None = None) -> Any`
  - Multiply ``samples`` by ``window`` element wise using the selected array backend.
- `detrend(values: Sequence[float], *, xp_module: Any | None = None) -> Any`
  - Remove the arithmetic mean from ``values`` without forcing a NumPy materialisation.
- `power_spectrum(samples: Sequence[float], sample_rate: float, *, xp_module: Any | None = None) -> List[Tuple[float, float]]`
  - Return the single-sided power spectrum of ``samples`` for the chosen backend.
- `cross_spectrum(input_series: Sequence[float], response_series: Sequence[float], sample_rate: float, *, xp_module: Any | None = None) -> List[Tuple[float, float, float]]`
  - Return the cross-spectrum between ``input_series`` and ``response_series`` while honouring the active backend.
- `phase_alignment(records: Sequence[SupportsTelemetrySample], *, steer_series: Iterable[float] | None = None, response_series: Iterable[float] | None = None, dominant_strategy: Literal['auto', 'fft', 'goertzel'] = 'auto', candidate_frequencies: Sequence[float] | None = None) -> Tuple[float, float, float]`
  - Estimate the dominant frequency and phase lag between steer and response.

Parameters
----------
records:
    Telemetry samples from which the sampling frequency is extracted.
steer_series, response_series:
    Explicit sequences containing steer input and the combined chassis
    response.  When omitted they are derived from ``records``.
dominant_strategy:
    ``"auto"`` uses Goertzel when SciPy is available and falls back to the
    FFT cross-spectrum otherwise. ``"goertzel"`` enforces the Goertzel path
    and raises when SciPy is unavailable, while ``"fft"`` always evaluates
    the dense FFT grid.
candidate_frequencies:
    Optional override for the candidate bins evaluated by the Goertzel path.

