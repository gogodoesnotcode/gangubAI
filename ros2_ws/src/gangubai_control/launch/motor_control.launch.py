"""
Launch motor controller with optional Gazebo simulation.

Usage:
  # Gazebo simulation with demo:
  ros2 launch gangubai_control motor_control.launch.py simulate:=true demo:=true

  # Hardware mode (Raspberry Pi):
  ros2 launch gangubai_control motor_control.launch.py
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_control = get_package_share_directory('gangubai_control')
    pkg_description = get_package_share_directory('gangubai_description')
    default_params = os.path.join(pkg_control, 'config', 'motor_params.yaml')

    # Read URDF for robot_state_publisher
    urdf_file = os.path.join(pkg_description, 'urdf', 'gangubai.urdf')
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    simulate_arg = DeclareLaunchArgument(
        'simulate', default_value='false',
        description='Launch Gazebo simulation (no GPIO)',
    )
    demo_arg = DeclareLaunchArgument(
        'demo', default_value='false',
        description='Run a demo sequence (forward/right/left/stop)',
    )
    params_arg = DeclareLaunchArgument(
        'params_file', default_value=default_params,
        description='Path to motor controller parameter file',
    )

    # ── Gazebo (only when simulate:=true) ────────────────────────────
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'),
                         'launch', 'gazebo.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('simulate')),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
        condition=IfCondition(LaunchConfiguration('simulate')),
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description',
                   '-entity', 'gangubai',
                   '-x', '0.0', '-y', '0.0', '-z', '1.0'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('simulate')),
    )

    # ── Motor controller node ────────────────────────────────────────
    motor_controller = Node(
        package='gangubai_control',
        executable='motor_controller',
        name='motor_controller',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'simulate': LaunchConfiguration('simulate'),
                'demo': LaunchConfiguration('demo'),
            },
        ],
    )

    return LaunchDescription([
        simulate_arg,
        demo_arg,
        params_arg,
        gazebo,
        robot_state_publisher,
        spawn_entity,
        motor_controller,
    ])
