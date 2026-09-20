import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/venus/projects/embodied-intelligence/ros2_ws/install/tb3_learning'
