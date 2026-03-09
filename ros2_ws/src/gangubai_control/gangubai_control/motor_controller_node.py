#!/usr/bin/env python3
"""
Motor controller node for a 2-wheel differential drive robot.
Controls 2 DC motors via an L298N H-bridge module on Raspberry Pi GPIO.

Modes:
  Hardware  – drives GPIO pins on Raspberry Pi via L298N
  Simulate  – publishes cmd_vel Twist for Gazebo diff_drive plugin

Subscriptions:
  /cmd_vel (geometry_msgs/Twist)   - velocity commands (hardware mode only)
  /motor_command (std_msgs/String) - simple text: forward, backward, left, right, stop

Publishers:
  /cmd_vel (geometry_msgs/Twist)   - velocity output (simulation mode only)
  /motor_status (std_msgs/String)  - current motor state

L298N wiring (default GPIO pins, configurable via ROS params):
  Motor A (left):  ENA=12  IN1=23  IN2=24
  Motor B (right): ENB=13  IN3=27  IN4=22
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

# Try importing RPi.GPIO — falls back to a stub for desktop development/testing
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    GPIO_AVAILABLE = False


class MotorControllerNode(Node):
    """ROS 2 node that drives two DC motors through an L298N module."""

    def __init__(self):
        super().__init__('motor_controller')

        # ── Declare parameters (all configurable at launch) ──────────────
        self.declare_parameter('ena_pin', 12)   # PWM pin for left motor
        self.declare_parameter('in1_pin', 23)
        self.declare_parameter('in2_pin', 24)
        self.declare_parameter('enb_pin', 13)   # PWM pin for right motor
        self.declare_parameter('in3_pin', 27)
        self.declare_parameter('in4_pin', 22)
        self.declare_parameter('pwm_freq', 1000)          # Hz
        self.declare_parameter('max_speed', 100.0)         # duty-cycle %
        self.declare_parameter('default_speed', 70.0)      # duty-cycle % for string commands
        self.declare_parameter('linear_scale', 0.3)        # m/s per 100% speed (for sim Twist)
        self.declare_parameter('angular_scale', 1.0)       # rad/s per 100% speed (for sim Twist)
        self.declare_parameter('cmd_vel_timeout', 0.5)     # seconds without cmd_vel → stop
        self.declare_parameter('simulate', not GPIO_AVAILABLE)  # force simulation mode
        self.declare_parameter('demo', False)                   # run a demo sequence on startup

        # Read parameters
        self.ena_pin = self.get_parameter('ena_pin').value
        self.in1_pin = self.get_parameter('in1_pin').value
        self.in2_pin = self.get_parameter('in2_pin').value
        self.enb_pin = self.get_parameter('enb_pin').value
        self.in3_pin = self.get_parameter('in3_pin').value
        self.in4_pin = self.get_parameter('in4_pin').value
        self.pwm_freq = self.get_parameter('pwm_freq').value
        self.max_speed = self.get_parameter('max_speed').value
        self.default_speed = self.get_parameter('default_speed').value
        self.linear_scale = self.get_parameter('linear_scale').value
        self.angular_scale = self.get_parameter('angular_scale').value
        self.cmd_vel_timeout = self.get_parameter('cmd_vel_timeout').value
        self.simulate = self.get_parameter('simulate').value
        self.demo = self.get_parameter('demo').value

        # ── GPIO setup (hardware mode only) ──────────────────────────────
        if not self.simulate:
            self._setup_gpio()
        else:
            self.pwm_a = None
            self.pwm_b = None

        # ── Subscribers ──────────────────────────────────────────────────
        # In hardware mode: subscribe to cmd_vel and convert to GPIO
        # In simulation mode: we PUBLISH cmd_vel for the Gazebo diff_drive plugin
        if not self.simulate:
            self.create_subscription(Twist, 'cmd_vel', self._cmd_vel_cb, 10)
        else:
            self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.create_subscription(String, 'motor_command', self._motor_command_cb, 10)

        # ── Status publisher ─────────────────────────────────────────────
        self.status_pub = self.create_publisher(String, 'motor_status', 10)

        # ── Safety timer: stop motors if no command received ─────────────
        self._last_cmd_time = self.get_clock().now()
        self._watchdog_timer = self.create_timer(0.1, self._watchdog_cb)
        self._stopped = True

        mode = 'SIMULATION (Gazebo)' if self.simulate else 'HARDWARE (RPi.GPIO)'
        self.get_logger().info(
            f'Motor controller started in {mode} mode'
        )
        if not self.simulate:
            self.get_logger().info(
                f'  GPIO pins: ENA={self.ena_pin} IN1={self.in1_pin} IN2={self.in2_pin} | '
                f'ENB={self.enb_pin} IN3={self.in3_pin} IN4={self.in4_pin}'
            )

        # ── Demo mode (timer-based so Gazebo can update) ─────────────────
        if self.demo:
            self._demo_steps = [
                ('FORWARD',  3.0),
                ('RIGHT',    2.0),
                ('LEFT',     2.0),
                ('FORWARD',  3.0),
                ('STOP',     0.0),
            ]
            self._demo_index = 0
            self._demo_elapsed = 0.0
            self.get_logger().info('═' * 50)
            self.get_logger().info('  DEMO MODE — running movement sequence')
            self.get_logger().info('═' * 50)
            self._start_demo_step()
            self._demo_timer = self.create_timer(0.1, self._demo_tick_cb)

    # ═══════════════════════════════════════════════════════════════════
    # GPIO helpers
    # ═══════════════════════════════════════════════════════════════════

    def _setup_gpio(self):
        """Initialise GPIO pins and PWM channels (hardware mode only)."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        pins = [
            self.ena_pin, self.in1_pin, self.in2_pin,
            self.enb_pin, self.in3_pin, self.in4_pin,
        ]
        for pin in pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

        self.pwm_a = GPIO.PWM(self.ena_pin, self.pwm_freq)
        self.pwm_b = GPIO.PWM(self.enb_pin, self.pwm_freq)
        self.pwm_a.start(0)
        self.pwm_b.start(0)

    def _set_motors(self, left_speed: float, right_speed: float):
        """
        Set individual motor speeds.
        Speed range: -100.0 (full reverse) … 0 (stop) … +100.0 (full forward).

        In simulation mode this publishes a Twist on /cmd_vel for Gazebo.
        In hardware mode this drives GPIO pins.
        """
        left_speed = max(-self.max_speed, min(self.max_speed, left_speed))
        right_speed = max(-self.max_speed, min(self.max_speed, right_speed))

        if self.simulate:
            self._publish_twist(left_speed, right_speed)
        else:
            self._apply_motor('left', left_speed)
            self._apply_motor('right', right_speed)

        # Publish status
        status = String()
        status.data = f'L={left_speed:+.1f}% R={right_speed:+.1f}%'
        self.status_pub.publish(status)

        self._stopped = (left_speed == 0.0 and right_speed == 0.0)

    def _publish_twist(self, left_speed: float, right_speed: float):
        """
        Convert left/right motor percentages to a Twist message and publish.
        The Gazebo diff_drive plugin will pick this up and move the robot.
        """
        # Recover linear/angular from differential speeds
        avg = (left_speed + right_speed) / 2.0 / self.max_speed  # -1..+1
        diff = (right_speed - left_speed) / 2.0 / self.max_speed  # -1..+1

        twist = Twist()
        twist.linear.x = avg * self.linear_scale
        twist.angular.z = diff * self.angular_scale
        self.cmd_vel_pub.publish(twist)

        self.get_logger().info(
            f'  [SIM] cmd_vel → linear={twist.linear.x:+.2f} m/s  '
            f'angular={twist.angular.z:+.2f} rad/s'
        )

    def _apply_motor(self, side: str, speed: float):
        """Drive one motor at the given signed speed via GPIO (-100…+100)."""
        if side == 'left':
            in1, in2, pwm = self.in1_pin, self.in2_pin, self.pwm_a
        else:
            in1, in2, pwm = self.in3_pin, self.in4_pin, self.pwm_b

        if speed > 0:
            GPIO.output(in1, GPIO.HIGH)
            GPIO.output(in2, GPIO.LOW)
        elif speed < 0:
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.HIGH)
        else:
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.LOW)

        pwm.ChangeDutyCycle(abs(speed))

    def _stop_motors(self):
        """Convenience: stop both motors."""
        self._set_motors(0.0, 0.0)

    # ═══════════════════════════════════════════════════════════════════
    # Callbacks
    # ═══════════════════════════════════════════════════════════════════

    def _cmd_vel_cb(self, msg: Twist):
        """
        Convert Twist to differential drive (hardware mode only).
        linear.x  → forward/backward  (-1.0 … +1.0)
        angular.z → turning            (-1.0 … +1.0)
        """
        self._last_cmd_time = self.get_clock().now()

        linear = max(-1.0, min(1.0, msg.linear.x))
        angular = max(-1.0, min(1.0, msg.angular.z))

        # Differential mixing
        left_speed = (linear - angular) * self.max_speed
        right_speed = (linear + angular) * self.max_speed

        self._set_motors(left_speed, right_speed)

    def _motor_command_cb(self, msg: String):
        """Handle simple text commands (works in both modes)."""
        command = msg.data.strip().lower()
        self._execute_command(command)

    def _execute_command(self, command: str):
        """Execute a named movement command."""
        speed = self.default_speed

        if command == 'forward':
            self._set_motors(speed, speed)
            self.get_logger().info('▶ Command: FORWARD')
        elif command == 'backward':
            self._set_motors(-speed, -speed)
            self.get_logger().info('▶ Command: BACKWARD')
        elif command == 'left':
            self._set_motors(-speed, speed)
            self.get_logger().info('▶ Command: LEFT')
        elif command == 'right':
            self._set_motors(speed, -speed)
            self.get_logger().info('▶ Command: RIGHT')
        elif command == 'stop':
            self._stop_motors()
            self.get_logger().info('■ Command: STOP')
        else:
            self.get_logger().warn(f'Unknown command: "{command}"')
            return

        # Reset watchdog
        self._last_cmd_time = self.get_clock().now()

    def _watchdog_cb(self):
        """Stop motors if no command received within timeout."""
        elapsed = (self.get_clock().now() - self._last_cmd_time).nanoseconds / 1e9
        if elapsed > self.cmd_vel_timeout and not self._stopped:
            self.get_logger().info('Timeout — stopping motors')
            self._stop_motors()

    # ═══════════════════════════════════════════════════════════════════
    # Demo (timer-based so ROS + Gazebo keep updating)
    # ═══════════════════════════════════════════════════════════════════

    def _start_demo_step(self):
        """Begin the current demo step."""
        if self._demo_index >= len(self._demo_steps):
            return
        command, duration = self._demo_steps[self._demo_index]
        self.get_logger().info(f'\n▶ DEMO [{self._demo_index+1}/{len(self._demo_steps)}]: '
                               f'{command} ({duration}s)')
        self.get_logger().info('─' * 40)
        self._execute_command(command.lower())
        self._demo_elapsed = 0.0

    def _demo_tick_cb(self):
        """Timer callback that advances the demo sequence."""
        if self._demo_index >= len(self._demo_steps):
            return

        _, duration = self._demo_steps[self._demo_index]
        self._demo_elapsed += 0.1

        # Keep re-publishing so Gazebo keeps getting cmd_vel
        if self.simulate and not self._stopped:
            command = self._demo_steps[self._demo_index][0].lower()
            speed = self.default_speed
            if command == 'forward':
                self._set_motors(speed, speed)
            elif command == 'backward':
                self._set_motors(-speed, -speed)
            elif command == 'left':
                self._set_motors(-speed, speed)
            elif command == 'right':
                self._set_motors(speed, -speed)

        if self._demo_elapsed >= duration and duration > 0:
            self._demo_index += 1
            if self._demo_index < len(self._demo_steps):
                self._start_demo_step()
            else:
                self.get_logger().info('\n' + '═' * 50)
                self.get_logger().info('  DEMO COMPLETE')
                self.get_logger().info('═' * 50)
                self._demo_timer.cancel()

    # ═══════════════════════════════════════════════════════════════════
    # Cleanup
    # ═══════════════════════════════════════════════════════════════════

    def destroy_node(self):
        self.get_logger().info('Shutting down motor controller — cleaning up GPIO')
        self._stop_motors()
        if not self.simulate:
            self.pwm_a.stop()
            self.pwm_b.stop()
            GPIO.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
