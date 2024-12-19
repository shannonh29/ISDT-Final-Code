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

# # Create a dictionary to store ArUco ID and its 3D coordinates
# aruco_info = {}

# spatial_anchor_info = {}  # Create an empty dictionary to store ID and coordinates

# callibrated = False

# spatial_anchor_amt = 3

# transform_matrix = np.array([
#     [-1.19262745e-04, 2.46563136e-02, 1.17432795e-03, 5.41771205e-01],
#     [-1.21803837e-03, 3.81684420e-02, -1.54870388e-03,  8.37794030e-01],
#     [-2.36497620e-03, 2.22619336e-02, -6.52206713e-03, 4.87152798e-01],
# 	[0.0, 0.0, 0.0, 1.0]
# ])

transform_matrix = np.load("transformation_matrix.npy")


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
	print("Sent: ", msg)

def transform_point(point):
	"""
	Transforms a 3D point using a 4x4 transformation matrix.

	Parameters:
		transform_matrix (numpy.ndarray): A 4x4 transformation matrix.
		point (list or numpy.ndarray): A 3D point [x, y, z].

	Returns:
		numpy.ndarray: Transformed 3D point [x', y', z'].
	"""
	# Convert the point to homogeneous coordinates
	homogeneous_point = np.append(point, 1)  # [x, y, z] -> [x, y, z, 1]

	# Perform the transformation
	transformed_homogeneous = np.dot(homogeneous_point, transform_matrix)
	transformed_point = transformed_homogeneous[:3]

	#Convert back to Cartesian coordinates
	# if transformed_homogeneous[3] != 0:  # Avoid division by zero
	# 	transformed_point = transformed_homogeneous[:3] / transformed_homogeneous[3]
	# else:
	# 	transformed_point = transformed_homogeneous[:3]

	return transformed_point


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
	sock.connect((HOST, PORT))
	print("Connected")

	while True:
		try:
			# if(not callibrated):

			# 	if(len(spatial_anchor_info) != spatial_anchor_amt):
			# 		msg = receive(sock)
			# 		anchor_id = msg.get("id")
			# 		anchor_coordinates = [msg.get("x"), msg.get("y"), msg.get("z")]  # Retrieve coordinates
			# 		spatial_anchor_info[anchor_id] = anchor_coordinates

			# 		# for anchor_id, coordinates in spatial_anchor_info.items():
			# 		# 	print(f"Anchor ID: {anchor_id}")
			# 		# 	print(f"Coordinates X={coordinates[0]}, Y={coordinates[1]}, Z={coordinates[2]}")
			# 		# 	print()  # For better readability
				
			# 	elif(len(aruco_info) != spatial_anchor_amt):
			# 		# Wait for a coherent pair of frames: depth and color
			# 		frames = pipeline.wait_for_frames()
			# 		depth_frame = frames.get_depth_frame()
			# 		color_frame = frames.get_color_frame()
			# 		if not depth_frame or not color_frame:
			# 			continue

			# 		# Get depth intrinsics for 3D projection
			# 		depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

			# 		# Convert images to numpy arrays
			# 		depth_image = np.asanyarray(depth_frame.get_data())
			# 		color_image = np.asanyarray(color_frame.get_data())
			
			# 		#Aruco Detection
			# 		corners, ids, rejected = arucoDetector.detectMarkers(color_image)
			# 		color_image = cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

			# 		if ids is not None:
			# 			for i, marker_corners in enumerate(corners):
			# 				x_center = np.mean(marker_corners[0][:, 0])
			# 				y_center = np.mean(marker_corners[0][:, 1])

			# 				depth_at_center = depth_frame.get_distance(x_center, y_center)

			# 				marker_3d_coordinates = rs.rs2_deproject_pixel_to_point(
			# 					depth_intrinsics, [x_center, y_center], depth_at_center
			# 				)

			# 				aruco_id = int(ids[i][0])  # Convert ID to an integer
			# 				aruco_info[aruco_id] = [ #Creating the dictionary key + info or replacing the info
			# 					marker_3d_coordinates[0],  # X-coordinate
			# 					marker_3d_coordinates[1],  # Y-coordinate
			# 					marker_3d_coordinates[2],  # Z-coordinate
			# 				]


			# 				for aruco_id, coordinates in aruco_info.items():
			# 					print(f"ArUco ID: {aruco_id}")
			# 					print(f"Coordinates X={coordinates[0]}, Y={coordinates[1]}, Z={coordinates[2]}")
			# 					print()  # For better readability

			# 				print(f"Completed")

			# 	elif(len(spatial_anchor_info) == spatial_anchor_amt and not callibrated and len(aruco_info) == spatial_anchor_amt):
			# 		#now we will determine the transformation matrix
			# 		unity_points = np.array(list(spatial_anchor_info.values()))
			# 		realsense_points = np.array(list(aruco_info.values()))
				
			# 		# Add a column of 1s to RealSense points to account for translation
			# 		realsense_aug = np.hstack([realsense_points, np.ones((realsense_points.shape[0], 1))])  # (n x 4)
				
			# 		# Solve for the transformation matrix using least squares
			# 		T, residuals, rank, s = np.linalg.lstsq(realsense_aug, unity_points, rcond=None)
				
			# 		# Append [0, 0, 0, 1] to make it a 4x4 transformation matrix
			# 		transform_matrix = np.vstack([T.T, [0, 0, 0, 1]])

			# 		callibrated = True
				
			# 		print("Transformation Matrix (4x4):")
			# 		print(transform_matrix)

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
			color_image = cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

			if ids is not None:
				for i, marker_corners in enumerate(corners):
					x_center = np.mean(marker_corners[0][:, 0])
					y_center = np.mean(marker_corners[0][:, 1])

					depth_at_center = depth_frame.get_distance(x_center, y_center)

					marker_3d_coordinates = rs.rs2_deproject_pixel_to_point(
						depth_intrinsics, [x_center, y_center], depth_at_center
					)

					transformed_pt = transform_point(marker_3d_coordinates)

					# Convert the transformed point to a dictionary for JSON serialization
					msg = {
						"id": 1,
    					"x": transformed_pt[0],
    					"y": transformed_pt[1],
    					"z": transformed_pt[2]
					}

						#msg = transform_point(marker_3d_coordinates)
					send(sock, msg)

						#aruco_id = int(ids[i][0])  # Convert ID to an integer
						# aruco_info[aruco_id] = [ #Creating the dictionary key + info or replacing the info
						# 	marker_3d_coordinates[0],  # X-coordinate
						# 	marker_3d_coordinates[1],  # Y-coordinate
						# 	marker_3d_coordinates[2],  # Z-coordinate
						# ]


						# for aruco_id, coordinates in aruco_info.items():
						# 	print(f"ArUco ID: {aruco_id}")
						# 	print(f"Coordinates X={coordinates[0]}, Y={coordinates[1]}, Z={coordinates[2]}")
						# 	print()  # For better readability

					#print(f"Marker ID {ids}: 3D Coordinates: {marker_3d_coordinates[0], marker_3d_coordinates[1], marker_3d_coordinates[2]}")

					#coord_text = f"X: {marker_3d_coordinates[0]:.2f} Y: {marker_3d_coordinates[1]:.2f} Z: {marker_3d_coordinates[2]:.2f}"
					#cv2.putText(color_image, coord_text, (int(x_center), int(y_center)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

			# Apply colormap on depth image (image must be converted to 8-bit per pixel first)
			#depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

			#depth_colormap_dim = depth_colormap.shape
			#color_colormap_dim = color_image.shape

			# If depth and color resolutions are different, resize color image to match depth image for display
			#if depth_colormap_dim != color_colormap_dim:
				#resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
				#images = np.hstack((resized_color_image, depth_colormap))
			#else:
				#images = np.hstack((color_image, depth_colormap))
			# Show images
			#cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
			#cv2.imshow('RealSense', images)
			#cv2.waitKey(1)

			#send(sock, msg)

		except KeyboardInterrupt:
			pipeline.stop()
			exit()
		except:
			pass
