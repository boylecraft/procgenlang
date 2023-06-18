import numpy as np
import random
from typing import List
import matplotlib.pyplot as plt


# class Letter1:
#     def __init__(self, pixels: List[List[bool]]):
#         self.pixels = pixels
#
# def generate_letters1(N=26, i=32, j=32, branch_rate = 0.5, decay_rate = -0.25) -> List[Letter]:
#     def get_neighbours(x, y):
#         """Get valid pixel neighbours for a given pixel."""
#         neighbours = [(x-1, y-1), (x, y-1), (x+1, y-1),
#                       (x-1, y),             (x+1, y),
#                       (x-1, y+1), (x, y+1), (x+1, y+1)]
#         neighbours = [(x, y) for x, y in neighbours if 0 <= x < i and 0 <= y < j]
#         return neighbours
#
#     letters = []
#     for _ in range(N):
#         pixels = np.zeros((i, j), dtype=bool)
#         start_x, start_y = random.randint(0, i-1), random.randint(0, j-1)
#         pixels[start_x, start_y] = True
#         frontier = [(start_x, start_y)]
#
#         while frontier:
#             new_frontier = []
#             for x, y in frontier:
#                 neighbours = get_neighbours(x, y)
#                 random.shuffle(neighbours)
#
#                 # the number of branches decreases with each step
#                 num_branches = max(1, int(branch_rate * len(neighbours)))
#                 branch_rate *= (1 + decay_rate)
#
#                 for nx, ny in neighbours[:num_branches]:
#                     if not pixels[nx, ny]:
#                         pixels[nx, ny] = True
#                         new_frontier.append((nx, ny))
#
#             frontier = new_frontier
#         # letters.append(Letter(pixels))
#         letters.append(Letter(np.invert(pixels)))
#
#     return letters
#
#
# def generate_letters2(N=26, i=32, j=32, branch_rate=0.5, decay_rate=-0.25,
#                      start_i=0.5, start_j=0.5, jitter=0.05) -> List[Letter]:
#     letters = []
#     for _ in range(N):
#         pixels = np.zeros((i, j), dtype=bool)
#
#         # Add jitter to starting position
#         jitter_i = np.random.uniform(-jitter, jitter)
#         jitter_j = np.random.uniform(-jitter, jitter)
#
#         start_x = int(j * (start_j + jitter_j))
#         start_y = int(i * (start_i + jitter_i))
#
#         # Ensuring that the starting points are within the valid range
#         start_x = np.clip(start_x, 0, j - 1)
#         start_y = np.clip(start_y, 0, i - 1)
#
#         frontier = [(start_x, start_y)]
#         while frontier:
#             x, y = frontier.pop()
#             if not pixels[y][x]:
#                 pixels[y][x] = True
#                 for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]:
#                     nx, ny = x + dx, y + dy
#                     if (0 <= nx < j) and (0 <= ny < i) and (np.random.rand() < branch_rate):
#                         branch_rate *= (1 + decay_rate)
#                         frontier.append((nx, ny))
#         letters.append(Letter(pixels))
#     return letters
#
#
# def generate_letters3(N=26, i=32, j=32, branch_rate=0.5, decay_rate=-0.25,
#                      start_i=0.5, start_j=0.5, jitter=0.05) -> List[Letter]:
#     letters = []
#     for _ in range(N):
#         pixels = np.zeros((i, j), dtype=bool)
#
#         # Add jitter to starting position
#         jitter_i = np.random.uniform(-jitter, jitter)
#         jitter_j = np.random.uniform(-jitter, jitter)
#
#         start_x = int(j * (start_j + jitter_j))
#         start_y = int(i * (start_i + jitter_i))
#
#         # Ensuring that the starting points are within the valid range
#         start_x = np.clip(start_x, 0, j - 1)
#         start_y = np.clip(start_y, 0, i - 1)
#
#         frontier = [(start_x, start_y)]
#         current_branch_rate = branch_rate  # reset the branch rate for each letter
#         while frontier:
#             x, y = frontier.pop()
#             if not pixels[y][x]:
#                 pixels[y][x] = True
#                 for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]:
#                     nx, ny = x + dx, y + dy
#                     if (0 <= nx < j) and (0 <= ny < i) and (np.random.rand() < current_branch_rate):
#                         current_branch_rate *= (1 + decay_rate)
#                         frontier.append((nx, ny))
#         letters.append(Letter(pixels))
#     return letters
#

#
# def generate_letters4(N=26, i=32, j=32, branch_rate = 0.5, decay_rate = -0.25,
#                      start_i=0.5, start_j=0.5, jitter=0.05, prune_threshold=6) -> List[Letter]:
#     letters = []
#     for _ in range(N):
#         pixels = np.zeros((i, j), dtype=bool)
#
#         jitter_i = np.random.uniform(-jitter, jitter)
#         jitter_j = np.random.uniform(-jitter, jitter)
#
#         start_x = int(j * (start_j + jitter_j))
#         start_y = int(i * (start_i + jitter_i))
#
#         start_x = np.clip(start_x, 0, j-1)
#         start_y = np.clip(start_y, 0, i-1)
#
#         frontier = [(start_x, start_y)]
#         current_branch_rate = branch_rate
#         while frontier:
#             x, y = frontier.pop()
#             if not pixels[y][x]:
#                 pixels[y][x] = True
#                 for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (-1,1), (1,-1)]:
#                     nx, ny = x + dx, y + dy
#                     if (0 <= nx < j) and (0 <= ny < i) and (np.random.rand() < current_branch_rate):
#                         current_branch_rate *= (1 + decay_rate)
#                         frontier.append((nx, ny))
#         pixels = prune_pixels(pixels, prune_threshold)
#         letters.append(Letter(pixels))
#     return letters
#
# def generate_letters5(N=26, i=32, j=32, branch_rate = 0.5, decay_rate = -0.25,
#                      start_i=0.5, start_j=0.5, jitter=0.05, prune_threshold=4,
#                      direction_weight=0.5) -> List[Letter]:
#     letters = []
#     directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (-1,1), (1,-1)]
#     for _ in range(N):
#         pixels = np.zeros((i, j), dtype=bool)
#
#         jitter_i = np.random.uniform(-jitter, jitter)
#         jitter_j = np.random.uniform(-jitter, jitter)
#
#         start_x = int(j * (start_j + jitter_j))
#         start_y = int(i * (start_i + jitter_i))
#
#         start_x = np.clip(start_x, 0, j-1)
#         start_y = np.clip(start_y, 0, i-1)
#
#         frontier = [(start_x, start_y)]
#         last_direction = None
#         current_branch_rate = branch_rate
#         while frontier:
#             x, y = frontier.pop()
#             if not pixels[y][x]:
#                 pixels[y][x] = True
#                 weights = np.ones(len(directions)) * (1-direction_weight)/(len(directions)-1)
#                 if last_direction is not None:
#                     # Prioritize last direction using direction_weight
#                     weights[directions.index(last_direction)] = direction_weight
#                 for _ in range(8):
#                     direction_index = np.random.choice(8, p=weights/np.sum(weights))
#                     dx, dy = directions[direction_index]
#                     nx, ny = x + dx, y + dy
#                     if (0 <= nx < j) and (0 <= ny < i) and (np.random.rand() < current_branch_rate):
#                         current_branch_rate *= (1 + decay_rate)
#                         frontier.append((nx, ny))
#                         last_direction = (dx, dy)
#         pixels = prune_pixels(pixels, prune_threshold)
#         letters.append(Letter(pixels))
#     return letters
#
# def generate_letters6(N=26, i=32, j=32, branch_rate=0.5, decay_rate=-0.25,
#                      start_i=0.5, start_j=0.5, jitter=0.05, prune_threshold=4,
#                      direction_weight=0.5, line_rate=0.5, line_decay=-0.1,
#                      max_lines=5) -> List[Letter]:
#     letters = []
#     directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (-1,1), (1,-1)]
#     for _ in range(N):
#         pixels = np.zeros((i, j), dtype=bool)
#
#         jitter_i = np.random.uniform(-jitter, jitter)
#         jitter_j = np.random.uniform(-jitter, jitter)
#
#         start_x = int(j * (start_j + jitter_j))
#         start_y = int(i * (start_i + jitter_i))
#
#         start_x = np.clip(start_x, 0, j-1)
#         start_y = np.clip(start_y, 0, i-1)
#
#         frontier = [(start_x, start_y)]
#         last_direction = None
#         current_branch_rate = branch_rate
#         current_line_rate = line_rate
#         lines = 1
#         while frontier or (lines < max_lines and np.random.rand() < current_line_rate):
#             if not frontier:
#                 print(lines)
#                 lines += 1
#                 current_line_rate *= (1 + line_decay)
#                 jitter_i = np.random.uniform(-jitter, jitter)
#                 jitter_j = np.random.uniform(-jitter, jitter)
#                 start_x = int(j * (0.5 + jitter_j))
#                 start_y = int(i * (0.5 + jitter_i))
#                 start_x = np.clip(start_x, 0, j-1)
#                 start_y = np.clip(start_y, 0, i-1)
#                 frontier.append((start_x, start_y))
#             else:
#                 x, y = frontier.pop()
#                 if not pixels[y][x]:
#                     pixels[y][x] = True
#                     weights = np.ones(len(directions)) * (1-direction_weight)/(len(directions)-1)
#                     if last_direction is not None:
#                         # Prioritize last direction using direction_weight
#                         weights[directions.index(last_direction)] = direction_weight
#                     for _ in range(8):
#                         direction_index = np.random.choice(8, p=weights/np.sum(weights))
#                         dx, dy = directions[direction_index]
#                         nx, ny = x + dx, y + dy
#                         if (0 <= nx < j) and (0 <= ny < i) and (np.random.rand() < current_branch_rate):
#                             current_branch_rate *= (1 + decay_rate)
#                             frontier.append((nx, ny))
#                             last_direction = (dx, dy)
#         pixels = prune_pixels(pixels, prune_threshold)
#         letters.append(Letter(pixels))
#     return letters
def prune_pixels(pixels, threshold):
    i, j = pixels.shape
    pruned = np.copy(pixels)
    for y in range(i):
        for x in range(j):
            count = 0
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (-1,1), (1,-1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < j) and (0 <= ny < i) and pixels[ny][nx]:
                    count += 1
            if count >= threshold:
                pruned[y][x] = False
    return pruned


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
                     max_lines=5, color_lines=False, SEED = 1234) -> List[Letter]:
    letters = []

      # Change to any number you like
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
                        direction_index = np.random.choice(8, p=weights/np.sum(weights))
                        dx, dy = directions[direction_index]
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < j) and (0 <= ny < i) and (np.random.rand() < current_branch_rate):
                            current_branch_rate *= (1 + decay_rate)
                            frontier.append((nx, ny))
                            last_direction = (dx, dy)
        pixels = prune_pixels(pixels, prune_threshold)
        if not color_lines:
            # if not color_lines, convert all nonzero entries to 1 (boolean array)
            pixels = (pixels > 0).astype(np.int32)
        letters.append(Letter(pixels))
    return letters


from PIL import Image

# def make_letter_png1(letters: List[Letter], spacing: int = 10, filename: str = "letters.png"):
#     # assuming all letters are the same size
#     i, j = letters[0].pixels.shape
#     width = j * len(letters) + spacing * (len(letters) - 1)
#     height = i
#     image = Image.new('1', (width, height), color = 0)
#     for index, letter in enumerate(letters):
#         letter_image = Image.fromarray(letter.pixels)
#         image.paste(letter_image, (index * (j + spacing), 0))
#
#     # for index, letter in enumerate(letters):
#     #     letter_image = Image.fromarray(letter.pixels.astype(np.uint8), mode='1')
#     #     image.paste(letter_image, (index * (j + spacing), 0))
#     image.save(filename)
#
#
# from PIL import Image
#
# from PIL import Image
#
# def make_letter_png2(letters: List[Letter], spacing: int = 10, filename: str = "letters.png", spacer_color: tuple = (0, 200, 0, 200)):
#     # assuming all letters are the same size
#     i, j = letters[0].pixels.shape
#     width = j * len(letters) + spacing * (len(letters) - 1)
#     height = i
#     image = Image.new('RGBA', (width, height), color = spacer_color)  # red spacer color
#     for index, letter in enumerate(letters):
#         # convert binary image to grayscale then to RGBA, replace colors (black for letters, white for background)
#         letter_image = Image.fromarray(np.uint8(letter.pixels)*255)  # binary to grayscale
#         letter_image = letter_image.convert('RGBA')  # grayscale to RGBA
#         datas = letter_image.getdata()
#         newData = []
#         for item in datas:
#             # change all white (also shades of whites)
#             # pixels to white
#             if item[0] in list(range(200, 256)):
#                 newData.append((255, 255, 255, 255))  # white (also shades of whites) -> white
#             else:
#                 newData.append((0, 0, 0, 255))  # black -> black
#         letter_image.putdata(newData)
#         image.paste(letter_image, (index * (j + spacing), 0), mask=letter_image)
#     image.save(filename)
#
#
# def make_letter_png3(letters: List[Letter], spacing: int = 10, filename: str = "letters.png", color_lines=False):
#     if color_lines:
#         # Get the max_lines from the data
#         max_lines = np.max([np.max(letter.pixels) for letter in letters])
#         color_map = plt.cm.get_cmap('gray', max_lines)
#     letter_imgs = []
#     for letter in letters:
#         img = Image.new('RGB', (letter.pixels.shape[1], letter.pixels.shape[0]), "white")
#         pixels = img.load()  # create the pixel map
#         for i in range(img.size[1]):  # for every pixel:
#             for j in range(img.size[0]):
#                 if letter.pixels[i][j] > 0:
#                     if color_lines:
#                         color = color_map(letter.pixels[i][j] % max_lines)[:3]
#                         color = tuple(int(255 * c) for c in color)
#                     else:
#                         color = (0, 0, 0)
#                     pixels[j, i] = color
#         letter_imgs.append(img)
#
#     total_width = sum([img.width for img in letter_imgs]) + spacing * (len(letter_imgs) + 1)
#     max_height = max([img.height for img in letter_imgs])
#
#     new_img = Image.new('RGB', (total_width, max_height), "red")
#
#     x_offset = spacing
#     for img in letter_imgs:
#         new_img.paste(img, (x_offset, 0))
#         x_offset += img.width + spacing
#
#     new_img.save(filename)
def make_letter_png(letters: List[Letter], spacing: int = 10, filename: str = "letters.png", color_lines=False,
                    max_width: int = 1000):
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
                     max_lines=3, color_lines=False, SEED=1234 )
    # for idx,ell in enumerate(new_letters):
    #     print(f'{idx}: {str(ell)}')
    # new_letters = generate_letters(N=2, i=10, j=10, branch_rate=0.75, decay_rate=-0.05)
    make_letter_png(letters=new_letters, spacing=1, filename="new_letters.png",color_lines=False, max_width=480 )
    show_image("new_letters.png")

    print("hi")
