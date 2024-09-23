"""
DSC 20 Project
Name(s): Melany Jacobo Mendoza, Nanda Payyappilly
PID(s):  A17474597, A17275118
Sources: Lecture 17 Slides, Lecture 20 Slides
"""

import numpy as np
import os
from PIL import Image

NUM_CHANNELS = 3


# --------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Saves the given RGBImage instance to the given path
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


# --------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    Represents an image in RGB format
    """

    def __init__(self, pixels):
        """
        Initializes a new RGBImage object

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        """
        if not type(pixels) ==  list:
            raise TypeError()
        if not len(pixels) > 0:
            raise TypeError()
        elements = [type(r) == list for r in pixels]
        if False in elements:
            raise TypeError()
        e = [len(r) >= 1 for r in pixels]
        if False in e:
            raise TypeError()
        size_one = len(pixels[0])
        same_length = [len(r) == size_one for r in pixels]
        if False in same_length:
            raise TypeError()
        col_type = [type(c) == list for r in pixels for c in r ]
        if False in col_type:
            raise TypeError()
        size_col = [len(c) == 3 for r in pixels for c in r]
        if False in size_col:
            raise TypeError()
        intensity_int = [True if type(e) == int else False for r in pixels \
        for c in r for e in c ]
        if False in intensity_int:
            raise TypeError()
        intensity = [True if e >= 0 and e<= 255 else False for r in pixels \
        for c in r for e in c]
        if False in intensity:
            raise ValueError()
        self.pixels = pixels
        self.num_rows = len(self.pixels)
        self.num_cols = len(self.pixels[0])

    def size(self):
        """
        Returns the size of the image in (rows, cols) format

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        return self.num_rows, self.num_cols

    def get_pixels(self):
        """
        Returns a copy of the image pixel array

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        copy_list =  [[[e for e in element] for element in row]
        for row in self.pixels]
        return copy_list

    def copy(self):
        """
        Returns a copy of this RGBImage object

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        deep_copy = RGBImage(self.get_pixels())
        return deep_copy

    def get_pixel(self, row, col):
        """
        Returns the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        if not type(row) == int:
            raise TypeError()
        if not type(col) == int:
            raise TypeError()
        if row < 0:
            raise ValueError()
        if col < 0:
            raise ValueError()
        if row >= self.num_rows:
            raise ValueError()
        if col >= self.num_cols:
            raise ValueError()
        return tuple(self.pixels[row][col])


    def set_pixel(self, row, col, new_color):
        """
        Sets the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        if not type(row) == int:
            raise TypeError()
        if not type(col) == int:
            raise TypeError()
        if row < 0:
            raise ValueError()
        if col < 0:
            raise ValueError()
        if row >= self.num_rows:
            raise ValueError()
        if col >= self.num_cols:
            raise ValueError()
        if not type(new_color) == tuple:
            raise TypeError()
        if len(new_color) != 3:
            raise TypeError()
        elements = [type(e) == int for e in new_color]
        if False in elements:
            raise TypeError()
        element = [False if e > 255 else True for e in new_color]
        if False in element:
            raise ValueError()
        modify = self.pixels[row][col]
        for i in range(len(new_color)):
            if new_color[i] >= 0:
                modify[i] = new_color[i]
        


# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    Contains assorted image processing methods
    Intended to be used as a parent class
    """

    def __init__(self):
        """
        Creates a new ImageProcessingTemplate object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        self.cost = 0

    def get_cost(self):
        """
        Returns the current total incurred cost

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        return self.cost

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels) 
        >>> img_negate = img_proc.negate(img)
        >>> id(img) != id(img_negate) # Check for new RGBImage instance
        True
       
        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img = img_read_helper('img/test_image_32x32.png')                 # 2
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')  # 3
        >>> img_negate = img_proc.negate(img)                              # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/test_image_32x32_negate.png', img_negate)# 6
        """
        negated_pixel = [[[255 - val for val in pixel] for pixel in r] \
        for r in image.get_pixels()]
        return RGBImage(negated_pixel)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_gray.png')
        >>> img_gray = img_proc.grayscale(img)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/test_image_32x32_gray.png', img_gray)
        """
        g_modify = [[[sum(c) // 3] * 3 for c in r] for r in image.get_pixels()]
        return RGBImage(g_modify)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/test_image_32x32_rotate.png', img_rotate)
        """
        row_reverse = [row[::-1] for row in image.get_pixels()]
        col_reverse = row_reverse[::-1]
        return RGBImage(col_reverse)

    def get_average_brightness(self, image):
        """
        Returns the average brightness for the given image

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.get_average_brightness(img)
        86
        """
        pixel_avg = [sum(c) // 3 for r in image.get_pixels() for c in r]
        num_pix = image.num_rows * image.num_cols
        total_avg = sum(pixel_avg) // num_pix
        return total_avg


    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_adjusted.png')
        >>> img_adjust = img_proc.adjust_brightness(img, 75)
        >>> img_adjust.pixels == img_exp.pixels # Check adjust_brightness
        True
        >>> img_save_helper('img/out/test_image_32x32_adjusted.png', img_adjust)
        """
        if not type(intensity) == int:
            raise TypeError()
        if intensity > 255 or intensity < -255:
            raise ValueError()

        mod_intensity = [[[255 if intensity + e > 255 else 0 if \
        intensity + e < 0 else intensity + e for e in c] for c in r] \
        for r in image.get_pixels()]
        return RGBImage(mod_intensity)

    def blur(self, image):
        """
        Returns a new image with the pixels blurred

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_blur.png')
        >>> img_blur = img_proc.blur(img)
        >>> img_blur.pixels == img_exp.pixels # Check blur
        True
        >>> img_save_helper('img/out/test_image_32x32_blur.png', img_blur)
        """
        # YOUR CODE GOES HERE #
        pixels = image.get_pixels()
    
        new_pixels = []
    
        for i in range(image.num_rows):
            new_row = []
            for j in range(image.num_cols):
                red_values = []
                green_values = []
                blue_values = []
            
                for x in range(max(0, i-1), min(i+2, image.num_rows)):
                    for y in range(max(0, j-1), min(j+2, image.num_cols)):
                        red_values.append(pixels[x][y][0])
                        green_values.append(pixels[x][y][1])
                        blue_values.append(pixels[x][y][2])
            
                r_avg = sum(red_values) // len(red_values)
                g_avg = sum(green_values) // len(green_values)
                b_avg = sum(blue_values) // len(blue_values)
            
                new_row.append([r_avg, g_avg, b_avg])
        
            new_pixels.append(new_row)
    
        return RGBImage(new_pixels)


# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    Represents a standard tier of an image processor
    """

    def __init__(self):
        """
        Creates a new StandardImageProcessing object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        super().__init__()
        self.calls = 0

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')
        >>> img_negate = img_proc.negate(img)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        if self.calls > 0:
            self.calls -= 1
        else:
            self.cost += 5
        return super().negate(image)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        """
        if self.calls > 0:
            self.calls -= 1
        else:
            self.cost += 6
        return super().grayscale(image)

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image
        """
        if self.calls > 0:
            self.calls -= 1
        else:
            self.cost += 10
        return super().rotate_180(image)

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level
        """
        if self.calls > 0:
            self.calls -= 1
        else:
            self.cost += 1
        return super().adjust_brightness(image, intensity)

    def blur(self, image):
        """
        Returns a new image with the pixels blurred
        """
        if self.calls > 0:
            self.calls -= 1
        else:
            self.cost += 5
        return super().blur(image)

    def redeem_coupon(self, amount):
        """
        Makes the given number of methods calls free

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """
        if not isinstance(amount, int):
            raise TypeError()
        if amount <= 0:
            raise ValueError()
        self.calls += amount


# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    Represents a paid tier of an image processor
    """

    def __init__(self):
        """
        Creates a new PremiumImageProcessing object

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        super().__init__()
        self.cost = 50

    def tile(self, image, new_width, new_height):
        """
        Returns a new image with size new_width x new_height where the
        given image is tiled to fill the new space

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> new_width, new_height = 70, 70
        >>> img_exp = img_read_helper('img/exp/square_32x32_tile.png')
        >>> img_tile = img_proc.tile(img_in, new_width, new_height)
        >>> img_tile.pixels == img_exp.pixels # Check tile output
        True
        >>> img_save_helper('img/out/square_32x32_tile.png', img_tile)
        """
        if not isinstance(image, RGBImage):
            raise TypeError()
        if type(new_width) != int or type(new_height) != int:
            raise TypeError()
        if new_width <= image.size()[0] or new_height <= image.size()[1]:
            raise ValueError()

        original_height, original_width = image.num_rows, image.num_cols
        pixel_width = new_width - original_width
        pixel_height = new_height - original_height
        new_pixels = [[[0,0,0] for i in range(new_width)] for j in range(new_height)]

        for i in range(pixel_height):
            for j in range(pixel_width):
                new_pixels[i][j] = image.get_pixels()[i%original_height][j%original_width]
        return RGBImage(new_pixels)


    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        Returns a copy of the background image where the sticker image is
        placed at the given x and y position.

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (31, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/test_image_32x32_sticker.png', img_combined)
        """
        if not isinstance(sticker_image, RGBImage) or \
        not isinstance(background_image, RGBImage):
            raise TypeError()
        if sticker_image.num_cols > background_image.num_cols:
            raise ValueError()
        if sticker_image.num_rows > background_image.num_rows:
            raise ValueError()
        if not isinstance(x_pos, int) or not isinstance(y_pos, int):
            raise TypeError()
        if x_pos < 0 or y_pos < 0:
            raise ValueError()
        if x_pos + sticker_image.num_cols > background_image.num_cols or \
        y_pos + sticker_image.num_rows > background_image.num_rows:
            raise ValueError()

        new_pixels = background_image.get_pixels()
        for i in range(sticker_image.num_rows):
            for j in range(sticker_image.num_cols):
                new_pixels[y_pos+i][x_pos+j] = sticker_image.get_pixels()[i][j]
        return RGBImage(new_pixels)

        
    def edge_highlight(self, image):
        """
        Returns a new image with the edges highlighted

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_edge = img_proc.edge_highlight(img)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_edge.png')
        >>> img_exp.pixels == img_edge.pixels # Check edge_highlight output
        True
        >>> img_save_helper('img/out/test_image_32x32_edge.png', img_edge)
        """
        single_pixels = [[sum(pixel) // 3 for pixel in r] for r in image.get_pixels()]
        kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]

        e_pixels = []
        for i in range(len(single_pixels)):
            e_row = []
            for j in range(len(single_pixels[0])):
                e_value = 0
                for k in range(len(kernel)):
                    for l in range(len(kernel[0])):
                        if 0 <= i + k < len(single_pixels) and \
                        0 <= j + l < len(single_pixels[0]):
                            e_value += single_pixels[i + k][j + l] * kernel[k][l]
                if e_value > 255:
                    e_value = 255
                if e_value < 0:
                    e_value = 0
                e_row.append(e_value)
            e_pixels.append(e_row)

            e_image = [[[value]*3 for value in row] for row in e_pixels]
        return RGBImage(e_image)


# Part 5: Image KNN Classifier #
class ImageKNNClassifier:
    """
    Represents a simple KNNClassifier
    """

    def __init__(self, k_neighbors):
        """
        Creates a new KNN classifier object
        """
        self.k_neighbors = k_neighbors

    def fit(self, data):
        """
        Stores the given set of data and labels for later
        """
        self.data = data
        if len(data) < self.k_neighbors:
            raise ValueError()

    def distance(self, image1, image2):
        """
        Returns the distance between the given images

        >>> img1 = img_read_helper('img/steve.png')
        >>> img2 = img_read_helper('img/knn_test_img.png')
        >>> knn = ImageKNNClassifier(3)
        >>> knn.distance(img1, img2)
        15946.312896716909
        """
        if not isinstance(image1, RGBImage) or not isinstance(image2, RGBImage):
            raise TypeError()
        if image1.size() != image2.size():
            raise ValueError()

        pixels_1 = [v for r in image1.get_pixels() for c in r for v in c]
        pixels_2 = [v for r in image2.get_pixels() for c in r for v in c]

        sum_squared_diffs = sum(list(map(lambda p1, p2: (p1 - p2) ** 2, pixels_1, pixels_2)))
        e_distance = sum_squared_diffs ** 0.5

        return e_distance


    def vote(self, candidates):
        """
        Returns the most frequent label in the given list

        >>> knn = ImageKNNClassifier(3)
        >>> knn.vote(['label1', 'label2', 'label2', 'label2', 'label1'])
        'label2'
        """
        dct = {}
        self.candidates = candidates
        for candidate in candidates:
            if candidate not in dct:
                dct[candidate] = 1
            else:
                dct[candidate] += 1

        max_label = ''
        max_count = 0

        for key, value in dct.items():
            if value > max_count:
                max_label = key
                max_count = value

        return max_label

    def predict(self, image):
        """
        Predicts the label of the given image using the labels of
        the K closest neighbors to this image

        The test for this method is located in the knn_tests method below
        """
        if not self.data:
            raise ValueError()

        neighbors = [(self.distance(image, tup[0]), tup[1]) for tup in self.data]
        sorted_neighbors = sorted(neighbors, key = lambda x:x[0])
        last_neighbor = self.k_neighbors + 1
        top_neighbors = sorted_neighbors[0:last_neighbor]

        return self.vote([n[1] for n in top_neighbors])



def knn_tests(test_img_path):
    """
    Function to run knn tests

    >>> knn_tests('img/knn_test_img.png')
    'nighttime'
    """
    # Read all of the sub-folder names in the knn_data folder
    # These will be treated as labels
    path = 'knn_data'
    data = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        # Ignore non-folder items
        if not os.path.isdir(label_path):
            continue
        # Read in each image in the sub-folder
        for img_file in os.listdir(label_path):
            train_img_path = os.path.join(label_path, img_file)
            img = img_read_helper(train_img_path)
            # Add the image object and the label to the dataset
            data.append((img, label))

    # Create a KNN-classifier using the dataset
    knn = ImageKNNClassifier(5)

    # Train the classifier by providing the dataset
    knn.fit(data)

    # Create an RGBImage object of the tested image
    test_img = img_read_helper(test_img_path)

    # Return the KNN's prediction
    predicted_label = knn.predict(test_img)
    return predicted_label
