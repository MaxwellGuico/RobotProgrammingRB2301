"""Standard ROS 2 map and markers, supported by Foxy and Jazzy RViz."""
import math

import numpy as np
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from visualization_msgs.msg import Marker, MarkerArray


def publish_diagnostics(node, state):
    """Draw the observed map, route, robot footprint, goal and labelled carrot in RViz.
    Everything is sent as standard markers in one frame so Foxy needs no custom display plugin.
    """
    if not node.nav_settings['debug_visuals']:
        return
    now = node.sensor_time()
    if now-node.last_debug_time < node.nav_settings['debug_interval']:
        return
    node.last_debug_time = now
    if node.debug_publisher is None:
        node.debug_publisher = node.create_publisher(MarkerArray, '/fp_debug/markers', 2)
    stamp = node.get_clock().now().to_msg()
    markers = []

    def marker(name, kind, color, size, points=(), z=0.08):
        """Build one marker with the common frame, colour and lifetime, then add it to this update."""
        message = Marker()
        message.header.frame_id, message.header.stamp = 'fp_world', stamp
        message.ns, message.type, message.action = name, kind, Marker.ADD
        message.pose.orientation.w = 1.0
        message.color.r, message.color.g, message.color.b, message.color.a = (*color, 1.0)
        message.scale.x = size
        message.scale.y = message.scale.z = size
        message.lifetime = Duration(seconds=1.0).to_msg()
        message.points = [Point(x=float(p[0]), y=float(p[1]), z=z) for p in points]
        markers.append(message)
        return message

    if node.sensor_ready:
        marker('scan', Marker.POINTS, (0.0, 0.8, 1.0), 0.025, node.scan_points)
    yaw = math.radians(node.pose[2])
    robot = marker('robot', Marker.ARROW, (0.0, 1.0, 0.0), 0.25)
    robot.scale.y = robot.scale.z = 0.06
    robot.pose.position.x, robot.pose.position.y = map(float, node.pose[:2])
    robot.pose.orientation.z, robot.pose.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
    p = node.nav_settings
    corners = np.array(((-p['footprint_rear'], -p['footprint_half_width']),
                        (p['footprint_front'], -p['footprint_half_width']),
                        (p['footprint_front'], p['footprint_half_width']),
                        (-p['footprint_rear'], p['footprint_half_width']),
                        (-p['footprint_rear'], -p['footprint_half_width'])))
    rotation = np.array(((math.cos(yaw), -math.sin(yaw)), (math.sin(yaw), math.cos(yaw))))
    marker('footprint', Marker.LINE_STRIP, (0.0, 1.0, 0.0), 0.015, corners@rotation.T+node.pose[:2])
    label = marker('state', Marker.TEXT_VIEW_FACING, (1.0, 1.0, 1.0), 0.12)
    label.pose.position.x, label.pose.position.y = map(float, node.pose[:2])
    label.pose.position.z = 0.4
    label.text = state
    marker('route', Marker.LINE_STRIP, (1.0, 1.0, 0.0), 0.025, node.route)
    marker('frontiers', Marker.POINTS, (0.5, 0.0, 1.0), 0.045,
           [node.cell_world(cell) for group in node.frontier_groups for cell in group])
    for name, point, color in (('goal', node.goal, (1.0, 0.0, 0.0)),
                               ('carrot', node.carrot, (1.0, 0.5, 0.0))):
        if point is not None:
            ball = marker(name, Marker.SPHERE, color, 0.10)
            ball.pose.position.x, ball.pose.position.y = map(float, point)
            ball.pose.position.z = 0.12
    if node.carrot is not None:
        marker('carrot_line', Marker.LINE_STRIP, (1.0, 0.5, 0.0), 0.02, (node.pose[:2], node.carrot))
        caption = marker('carrot_label', Marker.TEXT_VIEW_FACING, (1.0, 0.5, 0.0), 0.14)
        caption.pose.position.x, caption.pose.position.y = map(float, node.carrot)
        caption.pose.position.z, caption.text = 0.3, 'CARROT'
    # A marker map uses the same Foxy/Jazzy plugin as the route and avoids
    # the indexed-texture shader used by RViz's separate Map display.
    for value, name, color in ((0, 'free_space', (.45, .48, .52)),
                                (100, 'obstacles', (.9, .2, .15))):
        marker(name, Marker.POINTS, color, node.nav_settings['map_resolution'],
               [node.cell_world(cell) for cell, occupied in node.observation_grid.items() if occupied == value], z=0.0)
    node.debug_publisher.publish(MarkerArray(markers=markers))
