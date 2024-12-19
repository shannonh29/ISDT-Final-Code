"""
SIMPLE CLIENT FOR SOCKET CLIENT
"""

import socket
import json
import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Create a dictionary to store ArUco ID and its 3D coordinates
aruco_info = {}

spatial_anchor_info = {}  # Create an empty dictionary to store ID and coordinates

callibrated = False

spatial_anchor_amt = 5

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

#Setup ArUco Detector
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
arucoParams = cv2.aruco.DetectorParameters()
arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

# Start streaming
pipeline.start(config)

HOST = "192.168.1.146"  # The server's hostname or IP address
PORT = 80            # The port used by the server

def receive(sock):
    data = sock.recv(1024)
    data = data.decode('utf-8')
    msg = json.loads(data)
    print("Received: ", msg)
    return msg

def send(sock, msg):
	data = json.dumps(msg)
	sock.sendall(data.encode('utf-8'))
	#print("Sent: ", msg)




with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
	sock.connect((HOST, PORT))
	print("Connected")

	while (not callibrated):
		try:

			if(len(spatial_anchor_info) != spatial_anchor_amt):
				msg = receive(sock)
				anchor_id = msg.get("id")
				anchor_coordinates = [msg.get("x"), msg.get("y"), msg.get("z")]  # Retrieve coordinates
				spatial_anchor_info[anchor_id] = anchor_coordinates

				# for anchor_id, coordinates in spatial_anchor_info.items():
				# 	print(f"Anchor ID: {anchor_id}")
				# 	print(f"Coordinates X={coordinates[0]}, Y={coordinates[1]}, Z={coordinates[2]}")
				# 	print()  # For better readability

			elif(len(aruco_info) != spatial_anchor_amt):
				# Wait for a coherent pair of frames: depth and color
				frames = pipeline.wait_for_frames()
				depth_frame = frames.get_depth_frame()
				color_frame = frames.get_color_frame()
				if not depth_frame or not color_frame:
					continue

				# Get depth intrinsics for 3D projection
				depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

				# Convert images to numpy arrays
				depth_image = np.asanyarray(depth_frame.get_data())
				color_image = np.asanyarray(color_frame.get_data())
			
				#Aruco Detection
				corners, ids, rejected = arucoDetector.detectMarkers(color_image)
				print(f"ID: {ids}")
				color_image = cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

				if ids is not None:
					for i, marker_corners in enumerate(corners):
						x_center = np.mean(marker_corners[0][:, 0])
						y_center = np.mean(marker_corners[0][:, 1])

						depth_at_center = depth_frame.get_distance(x_center, y_center)

						marker_3d_coordinates = rs.rs2_deproject_pixel_to_point(
							depth_intrinsics, [x_center, y_center], depth_at_center
						)

						aruco_id = int(ids[i][0])  # Convert ID to an integer
						aruco_info[aruco_id] = [ #Creating the dictionary key + info or replacing the info
							marker_3d_coordinates[0],  # X-coordinate
							marker_3d_coordinates[1],  # Y-coordinate
							marker_3d_coordinates[2],  # Z-coordinate
						]


						for aruco_id, coordinates in aruco_info.items():
							print(f"ArUco ID: {aruco_id}")
							print(f"Coordinates X={coordinates[0]}, Y={coordinates[1]}, Z={coordinates[2]}")
							print()  # For better readability

						print(f"Completed")

			elif(len(spatial_anchor_info) == spatial_anchor_amt and len(aruco_info) == spatial_anchor_amt):
				#now we will determine the transformation matrix
				unity_points = np.array(list(spatial_anchor_info.values()))
				realsense_points = np.array(list(aruco_info.values()))
				
				# Add a column of 1s to RealSense points to account for translation
				realsense_aug = np.hstack([realsense_points, np.ones((realsense_points.shape[0], 1))])  # (n x 4)
				
				# Solve for the transformation matrix using least squares
				T, residuals, rank, s = np.linalg.lstsq(realsense_aug, unity_points, rcond=None)
				
				# Append [0, 0, 0, 1] to make it a 4x4 transformation matrix
				transformation_matrix = np.vstack([T.T, [0, 0, 0, 1]])
				
				print("Transformation Matrix (4x4):")
				print(transformation_matrix)
				# Save the transformation matrix as a .npy file
				np.save("transformation_matrix.npy", transformation_matrix)
				callibrated = True

			#send(sock, msg)
			
		except KeyboardInterrupt:
			pipeline.stop()
			exit()
		except:
			pass
