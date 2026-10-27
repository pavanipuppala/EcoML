"""
Unit tests for EcoML Deep Resource Telemetry and Carbon Tracking (Step 4).

Verifies process-level memory profiling (baseline, peak, delta), CPU utilization
sampling, sampling overhead measurement, and transparent CodeCarbon tracking
without fabricated metrics.
"""

import time
import pytest

from src.telemetry import (
    TelemetryCategory,
    ProcessTelemetrySnapshot,
    ProcessResourceMonitor,
    CarbonTelemetryResult,
    CarbonTrackerWrapper,
)


def test_process_resource_monitor_sampling():
    """Verify that ProcessResourceMonitor captures baseline, peak, and post memory."""
    monitor = ProcessResourceMonitor(sampling_interval_sec=0.01)
    monitor.start()

    # Allocate memory and perform some computation to simulate a workload
    dummy_data = [i ** 2 for i in range(200000)]
    time.sleep(0.04)

    snapshot = monitor.stop()

    assert isinstance(snapshot, ProcessTelemetrySnapshot)
    assert snapshot.baseline_ram_mb > 0.0
    assert snapshot.peak_ram_mb >= snapshot.baseline_ram_mb
    assert snapshot.post_ram_mb > 0.0
    assert snapshot.ram_delta_mb >= 0.0
    assert snapshot.logical_cpu_count >= 1
    assert snapshot.overhead_ms >= 0.0
    assert snapshot.samples_collected >= 1

    d = snapshot.to_dict()
    assert "baseline_ram_mb" in d
    assert "peak_ram_mb" in d
    assert "overhead_ms" in d


def test_carbon_tracker_disabled():
    """Verify that disabled carbon tracking transparently returns DISABLED status and None."""
    tracker = CarbonTrackerWrapper(enabled=False)
    tracker.start()
    time.sleep(0.01)
    res = tracker.stop()

    assert isinstance(res, CarbonTelemetryResult)
    assert res.energy_kwh is None
    assert res.carbon_emissions_g is None
    assert res.status == TelemetryCategory.DISABLED
    assert "Disabled" in res.status_message


def test_carbon_tracker_enabled_short_run():
    """Verify that a brief training run reports unavailable status without inventing values."""
    tracker = CarbonTrackerWrapper(enabled=True, country_iso_code="USA")
    tracker.start()
    # Microsecond execution
    res = tracker.stop()

    assert isinstance(res, CarbonTelemetryResult)
    # On a microsecond run, hardware counters either return None, 0.0 (marked unavailable), or exception
    if res.energy_kwh is None:
        assert res.status == TelemetryCategory.UNAVAILABLE
        assert res.carbon_emissions_g is None
    else:
        # If hardware supported sub-millisecond RAPL counter
        assert res.energy_kwh >= 0.0
        assert res.carbon_emissions_g is not None


def test_carbon_tracker_initialization_failure():
    """Verify that an invalid country code or failure does not crash the system."""
    tracker = CarbonTrackerWrapper(enabled=True, country_iso_code="INVALID_COUNTRY_XYZ")
    tracker.start()
    res = tracker.stop()
    assert isinstance(res, CarbonTelemetryResult)
    # Must report cleanly as unavailable rather than raising an unhandled exception
    assert res.status in (TelemetryCategory.UNAVAILABLE, TelemetryCategory.ESTIMATED)

