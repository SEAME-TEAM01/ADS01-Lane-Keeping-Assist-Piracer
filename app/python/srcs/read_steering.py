from adafruit_pca9685 import PCA9685
# from adafruit_motor import servo

class SteeringController:
    PWM_MAX_RAW_VALUE = 0xFFFF  # Assuming 16-bit PWM
    PWM_WAVELENGTH_50HZ = 1/50  # Wavelength for 50Hz
    PWM_STEERING_CHANNEL = 0    # Adjust according to your setup

    def __init__(self, i2c, channel=PWM_STEERING_CHANNEL):
        self.pwm_controller = PCA9685(i2c)
        self.pwm_controller.frequency = 50  # Set frequency to 50Hz
        self.channel = channel
        self.last_set_value = None

    def set_steering_percent(self, value):
        """Set and save the desired steering value in percent."""
        self._set_channel_active_time(self._get_50hz_duty_cycle_from_percent(-value), 
                                      self.pwm_controller, 
                                      self.channel)
        self.last_set_value = value  # Save the last set value

    def get_last_set_steering(self):
        """Retrieve the last set steering value."""
        return self.last_set_value

    @classmethod
    def _set_channel_active_time(cls, time, pwm_controller, channel):
        raw_value = int(cls.PWM_MAX_RAW_VALUE * (time / cls.PWM_WAVELENGTH_50HZ))
        pwm_controller.channels[channel].duty_cycle = raw_value

    @classmethod
    def _get_50hz_duty_cycle_from_percent(cls, percent):
        # Convert percentage to duty cycle time (adjust as needed)
        return percent * (cls.PWM_WAVELENGTH_50HZ / 2)

# Usage
import board
import busio

i2c = busio.I2C(board.SCL, board.SDA)
steering_controller = SteeringController(i2c)

steering_controller.set_steering_percent(300)
print("Last Set Steering:", steering_controller.get_last_set_steering())
