#!/usr/bin/env python
import rospy
import cv2
import apriltag
import numpy as np
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
class AprilTagNavigator:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('april_tag_nav', anonymous=True)

        # Create a publisher for the navigation goal
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.path_pub = rospy.Publisher('/scout_path', PoseArray, queue_size=10)

        self.odom_sub = rospy.Subscriber('/scout/odom', Odometry, self.odom_callback)

        # Initialize the camera (OpenCV)
        self.cap = cv2.VideoCapture(2)
        if not self.cap.isOpened():
            rospy.logerr("Error: Could not open camera.")
            return

        # Set up the AprilTag detector
        options = apriltag.DetectorOptions(families="tag36h11")
        self.detector = apriltag.Detector(options)

        # OpenCV bridge to convert ROS image messages
        self.bridge = CvBridge()

        rospy.loginfo("AprilTag Navigator Initialized.")
    
        # Camera calibration (replace with real values)
        self.FOCAL_LENGTH = 600  # Pixels
        self.TAG_SIZE = 0.16  # Meters

        # Robot pose (updated from /odom)
        self.robot_x = 0
        self.robot_y = 0
        self.robot_theta = 0

        self.pose_array = PoseArray()
        self.pose_array.header.frame_id = "odom" 
        self.pose_array.header.stamp = rospy.Time.now()


        self.current_goal = PoseStamped()
        self.robot_pose  = PoseStamped()

    def pose_distance(self, pose1, pose2):
        """Computes the Euclidean distance between two PoseStamped messages."""
        x1 = pose1.position.x 
        y1 = pose1.position.y
        x2 = pose2.position.x
        y2 = pose2.position.y
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def quaternion_to_euler(self, q):
        """ Converts a quaternion to Euler angles (roll, pitch, yaw) """
        euler = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return 0, 0, euler  # (roll, pitch, yaw)


    def odom_callback(self, msg):
        """ Updates the robot's current position from /odom """
        self.robot_pose = msg
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, _, self.robot_theta = self.quaternion_to_euler(orientation)

    def get_pitch_angle(self, detection):
        corners = detection.corners  # Detected tag corners

        left_edge = np.linalg.norm(corners[3] - corners[0])  # Left side
        right_edge = np.linalg.norm(corners[2] - corners[1]) # Right side

        # Compute yaw using the difference in perceived depth
        yaw_radians = np.arctan2(right_edge - left_edge, left_edge + right_edge)*-6
        yaw_degrees = np.degrees(yaw_radians)
        return yaw_radians, yaw_degrees

    def get_angle_to_tag(self, detection):
        distance = self.estimate_distance(detection)
        center = detection.center
        angle = np.arctan2((((300-center[0])/400)/distance), distance)
        return(angle)


    def estimate_distance(self, detection):
        tag_height_pixels = np.linalg.norm(detection.corners[3] - detection.corners[0])  # Left side height
        return (self.TAG_SIZE * self.FOCAL_LENGTH) / tag_height_pixels if tag_height_pixels > 0 else None


    def detect_tags(self):
        while not rospy.is_shutdown():
            # Read a frame from the camera
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn("Error: Could not read frame.")
                continue

            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect AprilTags in the frame
            detections = self.detector.detect(gray_frame)

            for detection in detections:

                distance = self.estimate_distance(detection)
                tag_id = detection.tag_id
                center = detection.center
                corners = detection.corners
                yaw_radians, yaw_degrees = self.get_pitch_angle(detection)

                angleToTag = self.get_angle_to_tag(detection)

                target_x = self.robot_x + (distance) * np.cos(angleToTag)
                target_y = self.robot_y + (distance) * np.sin(angleToTag)

                # rospy.loginfo("Detected AprilTag ID {}: Pitch = {:.2f} degrees".format(detection.tag_id, yaw_degrees))
    
                # Publish goal
                new_pose = Pose()
                new_pose.position.x = target_x
                new_pose.position.y = target_y
                new_pose.position.z = 0.0

                current_quaternion = self.robot_pose.pose.pose.orientation

                current_angle = euler_from_quaternion([current_quaternion.x,current_quaternion.y,current_quaternion.z,current_quaternion.w])
                print(current_angle)
                quaternion = quaternion_from_euler(0, 0, yaw_radians+current_angle[2])
                new_pose.orientation.x = quaternion[0]
                new_pose.orientation.y = quaternion[1]
                new_pose.orientation.z = quaternion[2]
                new_pose.orientation.w = quaternion[3]

                
                


                if not self.pose_array.poses or self.pose_distance(new_pose, self.pose_array.poses[-1]) > 0.2:
                    self.pose_array.poses.append(new_pose)

                if self.pose_array.poses:
                    if(self.current_goal.pose.position.x == 0 or self.pose_distance(self.robot_pose.pose.pose , self.current_goal.pose) < 0.1):
                        # print(self.pose_distance(self.robot_pose.pose , self.pose_array.poses[0]) )
                        new_goal = PoseStamped()
                        new_goal.pose = self.pose_array.poses.pop(0) 
                        new_goal.header.stamp = rospy.Time.now()
                        new_goal.header.frame_id = "odom"
                        self.current_goal = new_goal

                self.goal_pub.publish(self.current_goal)  
                self.path_pub.publish(self.pose_array)
                
                # rospy.loginfo("Navigating to AprilTag ID {}: x={:.2f}, y={:.2f}".format(tag_id, target_x, target_y))


                # Draw detections on the frame
                for corner in corners:
                    cv2.circle(frame, tuple(map(int, corner)), 5, (0, 255, 0), -1)
                cv2.circle(frame, tuple(map(int, center)), 5, (0, 0, 255), -1)
                cv2.putText(frame, "ID: {}".format(tag_id), (int(center[0]), int(center[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Show the frame
            cv2.imshow("AprilTag Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        node = AprilTagNavigator()
        node.detect_tags()
    except rospy.ROSInterruptException:
        pass
