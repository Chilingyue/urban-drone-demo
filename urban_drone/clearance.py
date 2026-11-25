"""SITL distances from vehicle reference point to observed obstacle surfaces.

New paths keep tracking reserve; already-flying paths and escape connectors
use separate, smaller execution limits. Unknown space still needs observation.
"""
VEHICLE_RADIUS = 0.35
EXECUTION_CLEARANCE = VEHICLE_RADIUS + 0.15
CONTACT_CLEARANCE = VEHICLE_RADIUS + 0.02
SEARCH_CLEARANCE = 1.20
PREFERRED_CLEARANCE = 1.80
CONNECT_MIN_CLEARANCE = EXECUTION_CLEARANCE + 0.03
INCUMBENT_CLEARANCE = EXECUTION_CLEARANCE + 0.10
RECOVERY_TRIGGER = SEARCH_CLEARANCE + 0.10
RECOVERY_TARGET = PREFERRED_CLEARANCE
MIN_APPROACH_SPEED = 0.20
# Escape must gain measurable clearance, not just surface-sampling ripple.
MIN_SEPARATION_SLOPE = 0.10
# Conservative outer-loop model, pending identification from SITL logs.
REACTION_TIME = 0.25
BRAKING_XY = 0.80
BRAKING_Z = 0.60
