"""
Cardiovascular Medication & Clinical Safety Subsystem
"""

try:
    from .medication_database import (
        CARDIOVASCULAR_MEDICATION_REPOSITORY,
        GLOBAL_MEDICATION_DB,
        MedicationDatabase,
        MedicationEntry,
        MedicationInteraction,
    )
    from .interaction_checker import (
        MedicationSafetyReport,
        SafetyAlert,
        check_medication_safety,
    )
except (ImportError, ValueError):
    from src.medications.medication_database import (
        CARDIOVASCULAR_MEDICATION_REPOSITORY,
        GLOBAL_MEDICATION_DB,
        MedicationDatabase,
        MedicationEntry,
        MedicationInteraction,
    )
    from src.medications.interaction_checker import (
        MedicationSafetyReport,
        SafetyAlert,
        check_medication_safety,
    )

__all__ = [
    "MedicationEntry",
    "MedicationInteraction",
    "MedicationDatabase",
    "GLOBAL_MEDICATION_DB",
    "CARDIOVASCULAR_MEDICATION_REPOSITORY",
    "SafetyAlert",
    "MedicationSafetyReport",
    "check_medication_safety",
]
