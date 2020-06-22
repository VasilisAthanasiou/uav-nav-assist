import cv2 as cv
import numpy as np
import os
from scipy import ndimage

from matplotlib import pyplot as plt

# ---------------------------------------------Image Processing--------------------------------------------------------- #

def process_image(src_img, resize=-1, rot_deg=0):
    '''Applies image processing techniques on an image.

    :param src_img: Image to be processed
    :param resize: Resize value. Takes float values between 0 and 1.
    :param rot_deg: Degrees that the image will be rotated
    :return: Returns the processed image

    '''

    # Resize image
    if resize != -1:
        res_img = cv.resize(src_img, None, fx=resize, fy=resize, interpolation=cv.INTER_CUBIC)

    # Convert to grayscale
    gray_image = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)

    processed_img = gray_image

    # Rotate the image
    if rot_deg != 0:
        processed_img = ndimage.rotate(gray_image, rot_deg)

    return processed_img

# ---------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------ Matching Algorithm -------------------------------------------------------- #

# Returns the top left pixel location of the sensed template
def find_target(src, temp):

    '''Uses openCV's matchTemplate to find a desired target inside the source image.

    :param src: Source image
    :param temp: Template image
    :return: Tuple containing the sensed target top left coordinates
    '''

    # Apply template matching
    res = cv.matchTemplate(src, temp, cv.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # Print the top-left position of the template inside the src
    print("TEMPLATE LOCATION = {}".format(max_loc))

    return max_loc  # Return top left position

# ---------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------ Statistical Analysis ------------------------------------------------------ #

def evaluate(src, temp, actual_match, n_templates, res_type, resize_value=-1, rotation=0, title=''):

    '''Performs the find_target() function for multiple source images on multiple templates, compares the results with
    the locations in the actual_match list, calculates the displacement and finally; writes the results on a text file.

    :param src: Source image
    :param temp: Template image
    :param actual_match: List of strings containing the correct coordinates for a give target
    :param n_templates: Number of template images
    :param res_type : Type of result to be return. Either 'text' or 'data'
    :param resize_value: Resize value for process_image()
    :param rotation: Rotation value for process_image()
    :return: A well structured string containing the results of the experiment or a tuple of the resulting data
    '''

    error = float(0)  # Error for the whole set
    error_img = []  # Error for each image Sum(error for every template / number of templates)
    num_of_images = len(src)

    counter = 0  # Keeps track of each iterations
    result = ''

    for img in src:
        img_height, img_width, _ = img.shape
        template_height, template_width, _ = temp[counter].shape

        processed_img = process_image(img)

        print("Source {}x{} , Template {}x{}".format(img_height, img_width, template_height, template_width))

        templates_per_image = n_templates  # This is the num of times the src will be scanned

        error_temp_img = 0  # Error for one template. How displaced the sensed frame is from the actual one

        for i in range(templates_per_image):
            # Store both actual and sensed x and y
            actual_x, actual_y, _ = actual_match[i].split(',')
            sensed_x, sensed_y = find_target(processed_img, process_image(temp[counter], rot_deg=rotation))

            # Error for one template is the added absolute differences between x and y divided the number of pixels
            error_temp_img += np.abs(int(sensed_x) - int(actual_x)) + np.abs(int(sensed_y) - int(actual_y))
            counter += 1
            print("For template {} : Sensed X : {} , Actual X : {} , Sensed Y : {} , Actual Y : {} , ERROR : {}".format(i, sensed_x, actual_x, sensed_y, actual_y, np.abs(int(sensed_x) - int(actual_x)) + np.abs(int(sensed_y) - int(actual_y))))


        error_img.append(error_temp_img / templates_per_image)  # Error for a whole image tested with multiple templates
        result += ("Mean error for image {} : {}px\n".format(int(counter / templates_per_image), error_temp_img / templates_per_image))

    # Error for all images
    error = sum(error_img)/num_of_images
    result += ("There is {} pixel mean error.\n\n".format(error))

    if res_type == 'text':
        return result
    if res_type == 'data':
        return ['{}'.format(counter + 1) for counter in range(len(error_img))], [round(error_img[counter], 2) for counter in range(len(error_img))], error, title
# ---------------------------------------------------------------------------------------------------------------------- #

# ----------------------------------------------- Simulation ----------------------------------------------------------- #
def simulate(sat_images, sim_uav_images, d_error=400, dx_bias='right', dy_bias='down', insertion_rot=45, capture_dim=200):
    """Runs a flight simulation.

    Args:
        sat_images: Set of satellite images, that represent images stored on the UAV
        sim_uav_images : Set of images that
        d_error: Displacement error. The error caused by the INS inaccuracy
        dx_bias: Direction of the horizontal error. Is always fixed for a give type of inertial system
        dy_bias: Direction of the vertical error. Is always fixed for a given type of inertial system
        insertion_rot: Inclination from which the UAV will be inserted into the satellite image.
        capture_dim: Dimension of image captured by the UAV
    Returns:

    """

    # Initialize variables
    displace_x = {'left': -d_error, 'right': d_error}
    displace_y = {'down': -d_error, 'up': d_error}
    heading = insertion_rot


    # Simulation loop
    for index in range(len(sat_images)):
        # Set center of satellite image
        sat_image_center = (int(sat_images[index].shape[0]/2), int(sat_images[index].shape[1]/2))
        print('Satellite image center : {}'.format(sat_image_center))

        # Process the UAV image
        uav_processed_image = cv.cvtColor(sim_uav_images[index], cv.COLOR_BGR2GRAY)
        cv.imshow('Gray image', uav_processed_image)  # Will probably produce error

        # Simulate heading errors
        inertial_error = np.random.uniform(0, 1)
        uav_processed_image = ndimage.rotate(uav_processed_image, heading + inertial_error)
        print('The INS made a {} degree error'.format(inertial_error))

        # Capturing and rotating image
        captured_img = uav_processed_image[displace_x[dx_bias]:displace_x[dx_bias]+capture_dim, displace_y[dy_bias]:displace_y[dy_bias]+capture_dim]
        cv.imshow('Captured image', captured_img)  # Will probably produce error
        while True:
            if cv.waitKey(1) == 27:
                cv.destroyAllWindows()
                break
        print('INS : {} degrees insertion angle\nRotating image accordingly...'.format(heading))
        captured_img = ndimage.rotate(captured_img, -heading)
        print('Captured ground image of size {}x{}'.format(captured_img.shape[0], captured_img.shape[1]))

        captured_img = captured_img[int(captured_img.shape[0]/4):int(captured_img.shape[0]/4) + int(captured_img.shape[0]/2), int(captured_img.shape[1]/4):int(captured_img.shape[1]/4) + int(captured_img.shape[1]/2)]
        cv.imshow('INS corrected captured image', captured_img)
        while True:
            if cv.waitKey(1) == 27:
                cv.destroyAllWindows()
                break
        # Find where the captured image is located relative to the satellite image
        captured_image_location = find_target(sat_images[index], captured_img)  # Top-left location of the template image
        print("DEBUG : captured_image_location={}".format(captured_image_location))
        # Define center of captured image inside the satellite image
        captured_image_center = (captured_image_location[0] + (capture_dim / 2), captured_image_location[1] + (capture_dim / 2))
        print("DEBUG : captured_image_center={}".format(captured_image_center))

        # Send the course correction signal
        course_displacement = sat_image_center[0] - captured_image_center[0], sat_image_center[1] - captured_image_center[1]
        print('The UAV is off center {} meters to the {} and {} meters {}'.format(course_displacement[0], dx_bias, course_displacement[1], dy_bias))


# ---------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------- Main --------------------------------------------------------------- #

# Set image directory
sat_directory = '../datasets/sources/source-diverse/1.source'
uav_directory = '../datasets/sources/source-diverse/4.blurred-cloudy'
# templates_directory = '../datasets/templates/templates-diverse/images'
# match_pos_path = '../datasets/templates/templates-diverse/dataset-diverse-loc.txt'
#
# # Read the txt file with the template's actual position
# match_pos_file = open(match_pos_path, 'r')
# actual_match_position = match_pos_file.readlines()

# Append all  the paths into lists
sat_paths = [os.path.join(sat_directory, image_path) for image_path in os.listdir(sat_directory)]
sat_paths.sort()

# Append UAV paths
uav_paths = [os.path.join(uav_directory, image_path) for image_path in os.listdir(uav_directory)]
uav_paths.sort()

# templates_paths = [os.path.join(templates_directory, template_path) for template_path in os.listdir(templates_directory)]
# templates_paths.sort(key=lambda name: int(os.path.splitext(os.path.basename(name))[0]))

# Read the satellite images
sat_images = []
gray_sat_images = []
for sat_path in sat_paths:
    sat_images.append(cv.cvtColor(cv.imread(sat_path), cv.COLOR_BGR2GRAY))
# Read the uav images
uav_images = []
for uav_path in uav_paths:
    uav_images.append(cv.imread(uav_path))

# Print all paths to make sure everything is ok
for elem in uav_paths:
    print(elem)

simulate(sat_images, uav_images)

# categories = ['Source', 'Blurred', 'Cloudy', 'Blurred and Cloudy']
# results = []
# counter = 0

# # Evaluate all variations of the source imagery
# for directory in source_paths:
#
#     for rot in range(1, 3):
#         file_path = [os.path.join(directory, image_path) for image_path in os.listdir(directory)]
#         file_path.sort()
#
#         # Read the images
#         source_images = []
#         for image_path in file_path:
#             source_images.append(cv.imread(image_path))
#
#         results.append(evaluate(source_images, templates, actual_match_position, 16, 'data', rotation=rot, title='Diverse Dataset {} with {} degree(s) rotation'.format(categories[counter], rot)))
#     counter += 1
#     source_images.clear()

    # Evaluate the matching method. The method is hardcoded into the evaluation. This should be changed
    # result_text = evaluate(source_images, templates, actual_match_position, 20, rotation=1)

    # Write the experiment results on a text file
    # file = open("../experiment-results.txt", "a")
    # file.write("-------------- Results using 1deg rotation on source images on less-features dataset --------------\n{}".format(result_text))

    # Draw plots

# print(results)
#
# colors = ['b', 'g', 'r', 'c', 'm', 'y', '#3277a8', '#a87332', '#915e49']
# counter = 0
# for result in results:
#     fig = plt.figure(counter)
#     plt.bar(result[0], result[1], color=colors[counter])
#     for index, value in enumerate(result[1]):
#         plt.text(index, value, str(value))
#
#     plt.xlabel('Images')
#     plt.ylabel('Mean pixel error')
#     plt.axis([0, 8, 0, 700])
#     ax = plt.gca()
#     ax.set_axisbelow(True)
#     plt.gca().yaxis.grid(linestyle="dashed")
#     plt.show()
#     counter += 1

# ---------------------------------------------------------------------------------------------------------------------- #
