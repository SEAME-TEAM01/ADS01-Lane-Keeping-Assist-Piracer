import math

def set_steering_radians(piracer=None, radians=1):
    if piracer is None:
        return
    max_steering_radians = math.radians(45)
    min_steering_radians = math.radians(-45)
    max_steering_percent = 1.0
    min_steering_percent = -1.0

    scaled_percent = ((radians - min_steering_radians) / (max_steering_radians - min_steering_radians)) * (max_steering_percent - min_steering_percent) + min_steering_percent

    piracer.set_steering_percent(scaled_percent)
