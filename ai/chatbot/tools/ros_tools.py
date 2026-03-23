"""ROS 2 motor-command tool – lets the chatbot move the robot."""

from typing import Literal

from langchain_core.tools import tool

# We use subprocess to publish a one-shot ROS 2 topic message.
# This avoids needing to spin up a full rclpy node inside the
# chatbot process — it just shells out to `ros2 topic pub`.
import subprocess

VALID_DIRECTIONS = ("forward", "backward", "left", "right", "stop")
VALID_WANDER_ACTIONS = ("start", "stop")


def _publish_once(topic: str, message_type: str, payload: str) -> None:
    """Publish one ROS message by shelling out to `ros2 topic pub --once`."""
    subprocess.Popen(
        [
            "ros2", "topic", "pub", "--once",
            topic, message_type,
            payload,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _publish_motor_command(direction: str, duration: float = 5.0) -> str:
    """Publish a string command to /motor_command and auto-stop after *duration* seconds."""
    try:
        # Send the movement command
        _publish_once(
            topic="/motor_command",
            message_type="std_msgs/msg/String",
            payload=f"{{data: '{direction}'}}",
        )

        # If the command is "stop" we're done — no need to schedule another stop
        if direction == "stop":
            return f"Robot stopped."

        # Schedule an automatic stop after `duration` seconds
        # (bash background sleep + ros2 pub)
        subprocess.Popen(
            ["bash", "-c",
             f"sleep {duration} && ros2 topic pub --once /motor_command "
             f"std_msgs/msg/String \"{{data: 'stop'}}\""],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return f"Robot moving {direction} for {duration}s then auto-stopping."

    except FileNotFoundError:
        return ("Error: `ros2` command not found. "
                "Make sure ROS 2 is sourced (source /opt/ros/humble/setup.bash).")
    except Exception as e:
        return f"Error sending motor command: {e}"


@tool
def move_robot(
    direction: Literal["forward", "backward", "left", "right", "360", "stop"],
    duration: float = 2.0,
) -> str:
    """Move the Gangubai robot in a given direction.

    Use this tool when the user asks the robot to move, drive, turn, spin,
    go forward/backward, rotate, or do a 360.

    Args:
        direction: The movement direction. One of:
            "forward"  – drive straight ahead
            "backward" – drive in reverse
            "left"     – turn/spin left
            "right"    – turn/spin right
            "360"      – do a full 360° clockwise spin
            "stop"     – immediately stop all motors
        duration: How long to move in seconds (default 2.0, ignored for "stop").
                  For "360" a good default is 3.0 seconds.

    Returns:
        A status message confirming the action.
    """
    direction = direction.strip().lower()

    # Handle the 360 spin: turn right for ~3 seconds (tune to your robot)
    if direction == "360":
        spin_duration = duration if duration != 2.0 else 10.0
        return _publish_motor_command("right", spin_duration)

    if direction not in VALID_DIRECTIONS:
        return (f"Unknown direction '{direction}'. "
                f"Valid options: forward, backward, left, right, 360, stop.")

    return _publish_motor_command(direction, duration)


@tool
def set_wander_mode(action: Literal["start", "stop"]) -> str:
    """Start or stop autonomous wander behavior.

    Use this tool when the user asks the robot to wander/explore autonomously,
    or to stop wandering and return to idle.

    Args:
        action: "start" to enter wander mode, "stop" to return to idle.

    Returns:
        A status message confirming the mode change.
    """
    action = action.strip().lower()

    if action not in VALID_WANDER_ACTIONS:
        return f"Invalid wander action '{action}'. Use start or stop."

    try:
        _publish_once(
            topic="/wander_mode",
            message_type="std_msgs/msg/String",
            payload=f"{{data: '{action}'}}",
        )
        if action == "start":
            return "Wander mode enabled. Robot will move autonomously until stopped."
        return "Wander mode stopped. Robot returned to idle."
    except FileNotFoundError:
        return (
            "Error: `ros2` command not found. "
            "Make sure ROS 2 is sourced (source /opt/ros/humble/setup.bash)."
        )
    except Exception as e:
        return f"Error changing wander mode: {e}"
