import numpy as np
import random
from typing import List
import matplotlib.pyplot as plt
from PIL import Image

def prune_pixels(pixels, threshold):
    '''
    Prunes pixels who have more than 'threshold' number of neighbors
    :param pixels: List of Lists of size i/j where 0 means no pixel.
    :param threshold: int: how many neighbors will cause this pixel to be deleted
    :return: pruned set of pixels
    '''
    ht_max, wid_max = pixels.shape
    pruned = np.copy(pixels)
    for y in range(ht_max):
        for x in range(wid_max):
            count = 0
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (-1,1), (1,-1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < wid_max) and (0 <= ny < ht_max) and pixels[ny][nx]:
                    count += 1
            if count >= threshold:
                pruned[y][x] = False
    return pruned

def inbounds(x,y,wid_max, ht_max,k):
    nx1, ny1, nx2, ny2 = x + k[0], y + k[1], x + k[2], y + k[3]
    if nx1 <0 or nx2 < 0 or ny1 < 0 or ny2 < 0:
        # print("out of bounds")
        return False
    if nx1 >= wid_max or nx2 >= wid_max or ny1 >= ht_max or ny2 >= ht_max:
        # print("out of bounds")
        return False
    # print("in bounds")
    # print(nx1, ny1, nx2, ny2)
    # print(x,y,wid_max, ht_max, k)
    return True
def connect_pixels(pixels):
    ht_max, wid_max = pixels.shape

    # Make a copy of the grid to avoid modifying it while iterating
    connected = np.copy(pixels)
    for y in range(ht_max):
        for x in range(wid_max):
            # If the current cell is empty...
            if pixels[y][x] == 0:
                for k in [(-1,0,1,0), (0,-1,0,1), (-1,-1,1,1), (1,1,-1,-1)]:
                    if inbounds(x,y,wid_max, ht_max, k):
                        if pixels[y+k[1]][x+k[0]] > 0 and pixels[y+k[3]][x+k[2]] > 0:
                            connected[y][x] = max(pixels[y+k[1]][x+k[0]], pixels[y+k[3]][x+k[2]])
    return connected




class Letter:
    def __init__(self, pixels):
        self.pixels = pixels

    def __str__(self):
        string = "-----\n"
        for idx,p in enumerate(self.pixels):
            string = f"{string}\n{idx} {p}"

        return string

def generate_letters(N=26, i=32, j=32, branch_rate=0.5, decay_rate=-0.25,
                     start_i=0.5, start_j=0.5, jitter=0.05, prune_threshold=4,
                     direction_weight=0.5, line_rate=0.5, line_decay=-0.1,
                     max_lines=5, color_lines=False, connect_lines=True, SEED = None) -> List[Letter]:
    '''

    This routine takes a bunch of inputs and applies really crude logic to try to create 'interesting' letters/characters.

    The approach is roughly:
    For each character (N)
        - pick a pixel (based on i, j, start_i, start_j, jitter)
        - pick the next pixel or stop (based on direction_weight, branch_rate, and decay_rate)
        - start a new line (based on max_lines, line_rate, and line_decay)
        - prune pixels (based on prune_threshold)

    Returns a List of Letter objects which represent the on/off pixels for each letter (based on color_lines).

    :param N: int: How many character/letters to create
    :param i: int: Image height for each letter
    :param j: int: Image width for each letter
    :param branch_rate: float: rate to set how long of a 'stroke' for each line
    :param decay_rate: float: rate to slow down branch_rate. should be negative. low values decay slower
    :param start_i: float: percentage value on where to 'start' first line in height direction. 0 is top of letter image. 0.5 means start at height/2
    :param start_j: float: percentage value on where to 'start' first line in widtch direction. 0 is left side of letter image. 0.33 means to start 1/3 from the left.
    :param jitter: float: how much to randomize start_i, start_j. 0 means no randomness.
    :param prune_threshold: float: part of post-processing step to delete a pixel if it has "prune_threshold" number of neighbors, including diagonal. This tries to remove 'blockiness'
                            i.e. if prune_threshold = 5, then any pixels with 5 neighbors or more will be deleted.
                            this logic isn't complex and always starts at the top-left of each letter image.
    :param direction_weight: float: if 1.0, the next pixel selected will almost always be in the same direction. if 0.0, the next pixel is in a random direction. The algorithm tries to adjust the weights in the opposite direction in a gradient.
                             i.e. if direction_weight = 0.8 and the last pixel selected was to the right of the previous one, there's a very high chance the next pixel will also be to the right (to extend the line). There's an extremely low chance for the next pixel to be to the left.
                             For other directions, the chance is weighted towards the Right direction. so Up-Right and Down-Right have higher weights than directly Up and Down (which have higher weights than Up-Left and Down-Left)
                             This isn't perfect.
    :param line_rate: float: How much of a chance to create an additional line after the previous line is done.
    :param line_decay: float: How fast the line_rate decays. should be negative. low values decay slower.
    :param max_lines: float: The max number of lines to create
    :param color_lines: bool: if True, will save data in a 'grayscale' where each subsequent line is lighter than the previous.
    :param SEED: Optional[int]: random seed. If not passed, script will always generate random values. set this value if you want repeatable results.
    :return: List[Letter]: list of Letter objects that can be passed to make_letter_png()
    '''
    letters = []

      # Change to any number you like
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
    print([random.random() for _ in range(5)], [np.random.random() for _ in range(5)])
    directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (-1,1), (1,-1)]
    for _ in range(N):
        # pixels is now an int array
        pixels = np.zeros((i, j), dtype=np.int32)

        jitter_i = np.random.uniform(-jitter, jitter)
        jitter_j = np.random.uniform(-jitter, jitter)

        start_x = int(j * (start_j + jitter_j))
        start_y = int(i * (start_i + jitter_i))

        start_x = np.clip(start_x, 0, j-1)
        start_y = np.clip(start_y, 0, i-1)

        frontier = [(start_x, start_y)]
        last_direction = None
        current_branch_rate = branch_rate
        current_line_rate = line_rate
        lines = 1
        while frontier or (lines < max_lines and np.random.rand() < current_line_rate):
            if not frontier:
                # print(f"newline: {lines}")
                lines += 1
                current_line_rate *= (1 + line_decay)
                jitter_i = np.random.uniform(-jitter, jitter)
                jitter_j = np.random.uniform(-jitter, jitter)
                start_x = int(j * (0.5 + jitter_j))
                start_y = int(i * (0.5 + jitter_i))
                start_x = np.clip(start_x, 0, j-1)
                start_y = np.clip(start_y, 0, i-1)
                frontier.append((start_x, start_y))
            else:
                x, y = frontier.pop()
                if pixels[y][x] == 0:
                    # assign the line number to pixel
                    pixels[y][x] = lines
                    weights = np.ones(len(directions)) * (1-direction_weight)/(len(directions)-1)
                    if last_direction is not None:
                        # Prioritize last direction using direction_weight
                        weights[directions.index(last_direction)] = direction_weight
                    for _ in range(8):
                        try:
                            direction_index = np.random.choice(8, p=weights/np.sum(weights))
                        except ValueError as e:
                            direction_index = random.randint(0,7)
                        dx, dy = directions[direction_index]
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < j) and (0 <= ny < i) and (np.random.rand() < current_branch_rate):
                            current_branch_rate *= (1 + decay_rate)
                            frontier.append((nx, ny))
                            last_direction = (dx, dy)
        if connect_lines:
            print("connecting")
            for connect in range(10):
                pixels = connect_pixels(pixels)
        pixels = prune_pixels(pixels, prune_threshold)
        if not color_lines:
            # if not color_lines, convert all nonzero entries to 1 (boolean array)
            pixels = (pixels > 0).astype(np.int32)
        letters.append(Letter(pixels))
    return letters

def make_letter_png(letters: List[Letter], spacing: int = 10, filename: str = "letters.png", color_lines=False,
                    max_width: int = 1000):
    '''

    :param letters: List of Letters
    :param spacing: int: how many pixels to space each letter in output png file
    :param filename: str: output png filename
    :param color_lines: bool: if True, expects Letters to be in 'grayscale' mode to identify each line/stroke and plots them in grayscale. if False, uses black
    :param max_width: int: max width of the output png. When printing letters, new rows will be created when a letter will make the image > max_width
    :return: nothing. saves the file to disk
    '''
    if color_lines:
        max_lines = np.max([np.max(letter.pixels) for letter in letters])
        color_map = plt.cm.get_cmap('gray', max_lines)

    letter_imgs = []
    for letter in letters:
        img = Image.new('RGB', (letter.pixels.shape[1], letter.pixels.shape[0]), "white")
        pixels = img.load()  # create the pixel map
        for i in range(img.size[1]):  # for every pixel:
            for j in range(img.size[0]):
                if letter.pixels[i][j] > 0:
                    if color_lines:
                        color = color_map(letter.pixels[i][j] % max_lines)[:3]
                        color = tuple(int(255 * c) for c in color)
                    else:
                        color = (0, 0, 0)
                    pixels[j, i] = color
        letter_imgs.append(img)

    # Calculate how many letters fit into one row
    letters_per_row = max_width // (letter_imgs[0].width + spacing)

    total_width = min(sum([img.width for img in letter_imgs[:letters_per_row]]) + spacing * (letters_per_row + 1),
                      max_width)
    max_height = max([img.height for img in letter_imgs])

    rows = int(np.ceil(len(letter_imgs) / letters_per_row))
    new_img = Image.new('RGB', (total_width, max_height * rows + spacing * (rows + 1)), "red")

    x_offset = spacing
    y_offset = spacing
    for idx, img in enumerate(letter_imgs):
        if (idx > 0 and idx % letters_per_row == 0):  # If we've filled the row
            y_offset += img.height + spacing
            x_offset = spacing
        new_img.paste(img, (x_offset, y_offset))
        x_offset += img.width + spacing

    new_img.save(filename)


def show_image(filename: str):
    Image.open(filename).show()

if __name__ == '__main__':
    new_letters = generate_letters(N=1, i=51, j=51, branch_rate=0.99, decay_rate=-0.01, start_i=0.5, start_j=0.5,
                                   jitter=0.1, prune_threshold=5, direction_weight=.80, line_rate=0.99, line_decay=-0.01,
                     max_lines=3, color_lines=False, connect_lines=True, SEED=1234 )

    make_letter_png(letters=new_letters, spacing=1, filename="new_letters.png",color_lines=False, max_width=480 )
    show_image("new_letters.png")

    print("hi")
