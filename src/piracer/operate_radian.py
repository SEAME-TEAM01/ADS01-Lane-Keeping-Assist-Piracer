import math

def set_steering_radians(piracer=None, radians=1):
    if piracer is None:
        return

    scaled_percent = radians / math.radians(45)

    scaled_percent = max(min(scaled_percent, 1.0), -1.0)

    piracer.set_steering_percent(scaled_percent)
