"""
EcoML — Deep Resource Telemetry and Energy/Carbon Tracking Module.

This module provides high-precision system resource telemetry using psutil and
isolated carbon/energy estimation via CodeCarbon. It tracks process-level baseline RAM,
peak observed RAM, RAM delta, CPU utilization, sampling overhead, and physical
vs. estimated energy measurements without fabricating data.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import threading
import time

import psutil


class TelemetryCategory(Enum):
    """Categorization of telemetry metrics indicating measurement fidelity."""
    MEASURED = "Measured (Direct Physical Sensor)"
    ESTIMATED = "Estimated (Hardware TDP/Architecture Model)"
    UNAVAILABLE = "Unavailable (Hardware Sensor Inactive or Duration Below Sampling Threshold)"
    DISABLED = "Disabled (Opt-In Telemetry Not Requested)"


@dataclass
class ProcessTelemetrySnapshot:
    """Stores resource consumption observations captured during a monitored execution.

    Attributes:
        baseline_ram_mb: Process resident set size (RSS) prior to workload execution.
        peak_ram_mb: Maximum resident set size observed across all background samples.
        post_ram_mb: Resident set size immediately following workload completion.
        ram_delta_mb: Net memory increment (max(0, post_ram - baseline_ram)).
        avg_cpu_percent: Mean process CPU utilization across workload duration.
        peak_cpu_percent: Peak process CPU utilization observed during workload execution.
        logical_cpu_count: Total available logical CPU cores on host machine.
        sampling_interval_sec: Polling interval used by background sampling thread.
        samples_collected: Total number of discrete polling snapshots recorded.
        overhead_ms: Measurement latency and thread coordination overhead in milliseconds.
    """
    baseline_ram_mb: float
    peak_ram_mb: float
    post_ram_mb: float
    ram_delta_mb: float
    avg_cpu_percent: float
    peak_cpu_percent: float
    logical_cpu_count: int
    sampling_interval_sec: float
    samples_collected: int
    overhead_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert telemetry snapshot into a clean dictionary."""
        return {
            "baseline_ram_mb": round(self.baseline_ram_mb, 2),
            "peak_ram_mb": round(self.peak_ram_mb, 2),
            "post_ram_mb": round(self.post_ram_mb, 2),
            "ram_delta_mb": round(self.ram_delta_mb, 2),
            "avg_cpu_percent": round(self.avg_cpu_percent, 2),
            "peak_cpu_percent": round(self.peak_cpu_percent, 2),
            "logical_cpu_count": self.logical_cpu_count,
            "sampling_interval_sec": self.sampling_interval_sec,
            "samples_collected": self.samples_collected,
            "overhead_ms": round(self.overhead_ms, 3),
        }


class ProcessResourceMonitor:
    """Threaded background resource monitor for fine-grained process profiling.

    Periodically samples resident memory and CPU utilization of the current Python
    process during model training or inference without blocking algorithm execution.
    """

    def __init__(self, sampling_interval_sec: float = 0.02) -> None:
        """Initialize the process monitor.

        Args:
            sampling_interval_sec: Sleep interval between polling samples (default: 20ms).
        """
        self.sampling_interval_sec = max(0.005, sampling_interval_sec)
        self.process = psutil.Process()
        self._stop_event = threading.Event()
        self._sampler_thread: Optional[threading.Thread] = None
        self._cpu_samples: List[float] = []
        self._peak_ram_bytes: int = 0
        self._baseline_ram_bytes: int = 0
        self._start_time: float = 0.0
        self._overhead_start: float = 0.0

    def start(self) -> None:
        """Capture baseline metrics and spawn background sampling thread."""
        self._overhead_start = time.perf_counter()
        try:
            self._baseline_ram_bytes = self.process.memory_info().rss
        except Exception:
            self._baseline_ram_bytes = 0
        self._peak_ram_bytes = self._baseline_ram_bytes
        self._cpu_samples.clear()
        self._stop_event.clear()

        # Prime psutil CPU percent calculation (first call always returns 0.0)
        try:
            self.process.cpu_percent(interval=None)
        except Exception:
            pass

        self._sampler_thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._sampler_thread.start()
        self._start_time = time.perf_counter()

    def _sampling_loop(self) -> None:
        """Target loop for background sampling thread."""
        while not self._stop_event.is_set():
            try:
                mem_rss = self.process.memory_info().rss
                cpu_p = self.process.cpu_percent(interval=None)
                if mem_rss > self._peak_ram_bytes:
                    self._peak_ram_bytes = mem_rss
                self._cpu_samples.append(cpu_p)
            except Exception:
                pass
            time.sleep(self.sampling_interval_sec)

    def stop(self) -> ProcessTelemetrySnapshot:
        """Stop background sampling and compute final telemetry snapshot.

        Returns:
            ProcessTelemetrySnapshot: Aggregated CPU and memory observations.
        """
        self._stop_event.set()
        if self._sampler_thread is not None and self._sampler_thread.is_alive():
            self._sampler_thread.join(timeout=1.0)

        # Final synchronous memory check
        try:
            post_ram_bytes = self.process.memory_info().rss
        except Exception:
            post_ram_bytes = self._peak_ram_bytes

        if post_ram_bytes > self._peak_ram_bytes:
            self._peak_ram_bytes = post_ram_bytes

        overhead_duration_ms = (time.perf_counter() - self._overhead_start) * 1000.0

        baseline_mb = self._baseline_ram_bytes / (1024 * 1024)
        peak_mb = self._peak_ram_bytes / (1024 * 1024)
        post_mb = post_ram_bytes / (1024 * 1024)
        ram_delta_mb = max(0.0, post_mb - baseline_mb)

        cpu_avg = float(sum(self._cpu_samples) / len(self._cpu_samples)) if self._cpu_samples else 0.0
        cpu_peak = float(max(self._cpu_samples)) if self._cpu_samples else cpu_avg

        logical_cpus = psutil.cpu_count(logical=True) or 1

        return ProcessTelemetrySnapshot(
            baseline_ram_mb=baseline_mb,
            peak_ram_mb=peak_mb,
            post_ram_mb=post_mb,
            ram_delta_mb=ram_delta_mb,
            avg_cpu_percent=cpu_avg,
            peak_cpu_percent=cpu_peak,
            logical_cpu_count=logical_cpus,
            sampling_interval_sec=self.sampling_interval_sec,
            samples_collected=len(self._cpu_samples),
            overhead_ms=overhead_duration_ms,
        )


@dataclass
class CarbonTelemetryResult:
    """Container for environmental emission measurements captured via CodeCarbon.

    Attributes:
        energy_kwh: Electrical consumption in kilowatt-hours, or None if unavailable.
        carbon_emissions_g: Carbon emissions in grams CO2eq, or None if unavailable.
        status: TelemetryCategory describing reliability and source.
        status_message: Informative explanation of measurement or unavailability reason.
    """
    energy_kwh: Optional[float]
    carbon_emissions_g: Optional[float]
    status: TelemetryCategory
    status_message: str


class CarbonTrackerWrapper:
    """Manages optional, isolated execution of CodeCarbon emission tracking.

    Ensures that tracker failures, unprivileged hardware states, and rapid runs
    are caught and reported transparently without substituting zeroes or fabricated numbers.
    """

    def __init__(self, enabled: bool = False, country_iso_code: str = "USA") -> None:
        """Initialize the carbon tracker wrapper.

        Args:
            enabled: Whether opt-in tracking is requested by the user.
            country_iso_code: Country ISO code for regional grid carbon intensity (default 'USA').
        """
        self.enabled = enabled
        self.country_iso_code = country_iso_code
        self._tracker: Optional[Any] = None
        self._init_error: Optional[str] = None

    def start(self) -> None:
        """Attempt to instantiate and start CodeCarbon tracker if enabled."""
        if not self.enabled:
            return

        try:
            from codecarbon import OfflineEmissionsTracker
            # Use offline tracker with designated country to avoid internet dependency
            self._tracker = OfflineEmissionsTracker(
                country_iso_code=self.country_iso_code,
                save_to_file=False,
                log_level="error",
            )
            self._tracker.start()
        except Exception as exc:
            self._tracker = None
            self._init_error = str(exc)

    def stop(self) -> CarbonTelemetryResult:
        """Stop tracking and process observed emissions.

        Returns:
            CarbonTelemetryResult: Validated emissions measurement or transparent unavailable status.
        """
        if not self.enabled:
            return CarbonTelemetryResult(
                energy_kwh=None,
                carbon_emissions_g=None,
                status=TelemetryCategory.DISABLED,
                status_message="Disabled (Opt-in tracking not requested)",
            )

        if self._tracker is None:
            reason = f"Tracker failed to initialize: {self._init_error}" if self._init_error else "Sensor unavailable"
            return CarbonTelemetryResult(
                energy_kwh=None,
                carbon_emissions_g=None,
                status=TelemetryCategory.UNAVAILABLE,
                status_message=f"Unavailable ({reason})",
            )

        try:
            emissions_kg = self._tracker.stop()
            if emissions_kg is not None and emissions_kg > 0:
                carbon_g = round(emissions_kg * 1000.0, 6)
                energy_kwh: Optional[float] = None
                if hasattr(self._tracker, "_total_energy") and hasattr(self._tracker._total_energy, "kWh"):
                    energy_kwh = round(float(self._tracker._total_energy.kWh), 8)

                return CarbonTelemetryResult(
                    energy_kwh=energy_kwh,
                    carbon_emissions_g=carbon_g,
                    status=TelemetryCategory.ESTIMATED,
                    status_message="Estimated (CodeCarbon Hardware Telemetry)",
                )
            else:
                return CarbonTelemetryResult(
                    energy_kwh=None,
                    carbon_emissions_g=None,
                    status=TelemetryCategory.UNAVAILABLE,
                    status_message="Unavailable (Training duration below physical sampling threshold)",
                )
        except Exception as exc:
            return CarbonTelemetryResult(
                energy_kwh=None,
                carbon_emissions_g=None,
                status=TelemetryCategory.UNAVAILABLE,
                status_message=f"Unavailable ({str(exc)})",
            )

