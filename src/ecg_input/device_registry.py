"""
ECG Device Compatibility & Validation Registry
===============================================

Catalogs known hospital ECG devices, acquisition systems, formats, and their
clinical validation status under CDSCO/MDR-2017 principles.

Hard Regulatory Principle:
Never claim compatibility or validation status with a hospital device
unless verified by explicit test evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class DeviceProfile:
    """Standardized profile for an ECG acquisition system or format specification."""
    device_id: str
    device_name: str
    manufacturer: str
    format: str
    expected_sampling_rates: List[float]
    lead_configurations: List[int]  # e.g. [1] or [12]
    units: str
    supported_status: bool
    validation_status: str  # BENCHMARK_VALIDATED, EXPERIMENTAL, UNTESTED, REJECTED
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DeviceRegistry:
    """Registry maintaining tested device profiles and format capabilities."""

    def __init__(self):
        self._registry: Dict[str, DeviceProfile] = {}
        self._populate_default_profiles()

    def _populate_default_profiles(self):
        # 1. MIT-BIH PhysioNet Benchmark Systems (Fully Verified)
        self.register(
            DeviceProfile(
                device_id="MIT_BIH_WFDB",
                device_name="MIT-BIH Ambulatory Holter System",
                manufacturer="Del Mar Avionics / PhysioNet",
                format="WFDB",
                expected_sampling_rates=[360.0],
                lead_configurations=[1, 2],
                units="mV",
                supported_status=True,
                validation_status="BENCHMARK_VALIDATED",
                notes="Used for training and offline benchmark evaluation. Modified Lead II.",
            )
        )

        # 2. Universal Delimited CSV/TXT/NPY Export (Tested)
        self.register(
            DeviceProfile(
                device_id="UNIVERSAL_CSV_DIGITAL",
                device_name="Standard Digital Delimited Export",
                manufacturer="Generic Clinical Exporter",
                format="CSV",
                expected_sampling_rates=[125.0, 250.0, 360.0, 500.0, 1000.0],
                lead_configurations=[1],
                units="mV",
                supported_status=True,
                validation_status="BENCHMARK_VALIDATED",
                notes="Tested with auto-delimiting and single-lead voltage isolation.",
            )
        )

        # 3. GE Healthcare MAC 5500 / 2000 (Planned / Untested)
        self.register(
            DeviceProfile(
                device_id="GE_MAC_XML",
                device_name="GE MAC 5500 / 2000 Rest ECG System",
                manufacturer="GE Healthcare",
                format="XML",
                expected_sampling_rates=[500.0],
                lead_configurations=[12],
                units="uV",
                supported_status=False,
                validation_status="UNTESTED",
                notes="Requires hospital XML schema validation. Not yet tested with local hardware.",
            )
        )

        # 4. Philips PageWriter TC Series (Planned / Untested)
        self.register(
            DeviceProfile(
                device_id="PHILIPS_PAGEWRITER_XML",
                device_name="Philips PageWriter TC70/TC50",
                manufacturer="Philips Healthcare",
                format="XML",
                expected_sampling_rates=[500.0],
                lead_configurations=[12],
                units="uV",
                supported_status=False,
                validation_status="UNTESTED",
                notes="Philips Sierra ECG XML format. Awaiting hospital hardware sample captures.",
            )
        )

        # 5. Standard DICOM 12-Lead Waveform (SOP Class 1.2.840.10008.5.1.4.1.1.9.1.1)
        self.register(
            DeviceProfile(
                device_id="DICOM_WAVEFORM_12LEAD",
                device_name="DICOM 12-Lead ECG Waveform Storage",
                manufacturer="NEMA / DICOM Standards Committee",
                format="DICOM",
                expected_sampling_rates=[500.0, 1000.0],
                lead_configurations=[12],
                units="uV",
                supported_status=False,
                validation_status="UNTESTED",
                notes="Standard hospital PACS interoperability target. Schema mapped, awaiting clinical dataset.",
            )
        )

        # 6. European Data Format (EDF / EDF+)
        self.register(
            DeviceProfile(
                device_id="EDF_PLUS_WAVEFORM",
                device_name="European Data Format (EDF+)",
                manufacturer="Kemp et al. / Open Standard",
                format="EDF",
                expected_sampling_rates=[200.0, 250.0, 500.0],
                lead_configurations=[1, 3, 12],
                units="uV",
                supported_status=False,
                validation_status="UNTESTED",
                notes="Target for polysomnography and telemetry recordings.",
            )
        )

    def register(self, profile: DeviceProfile):
        """Register a new device or format profile."""
        self._registry[profile.device_id] = profile

    def get(self, device_id: str) -> Optional[DeviceProfile]:
        """Retrieve profile by ID."""
        return self._registry.get(device_id)

    def list_all(self) -> List[DeviceProfile]:
        """List all registered device profiles."""
        return list(self._registry.values())

    def list_supported(self) -> List[DeviceProfile]:
        """List devices that are currently fully supported in software."""
        return [p for p in self._registry.values() if p.supported_status]

    def is_validated(self, device_id: str) -> bool:
        """Check whether a device format has completed benchmark/clinical validation."""
        profile = self._registry.get(device_id)
        return profile is not None and profile.validation_status == "BENCHMARK_VALIDATED"


# Global singleton instance
DEVICE_REGISTRY = DeviceRegistry()
