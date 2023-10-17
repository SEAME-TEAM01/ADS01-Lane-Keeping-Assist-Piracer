import math

def set_steering_radians(piracer=None, radian=1):
    if piracer is None:
        return

    max_radian = math.radians(135)
    min_radian = math.radians(45)

    if radian >= min_radian and radian <= max_radian:
        scaled_percent = (radian - min_radian) / (max_radian - min_radian) * 2 - 1
    else:
        scaled_percent = -1.0 if radian < min_radian else 1.0
    print(scaled_percent)
    # piracer.set_steering_percent(scaled_percent)
    # piracer.set_throttle_percent(0.5)
