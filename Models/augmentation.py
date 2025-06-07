import cv2
import numpy as np
import random
from skimage.draw import polygon

def generate_random_blob(num_points=100, max_deformation=0.05):
    angles = np.linspace(0, 2 * np.pi, num_points)
    base_radius = 1.0
    random_numbers = [random.uniform(0.5, 7) for _ in range(5)]
    deformation = (
        max_deformation * np.sin(random_numbers[0] * angles + 2 * np.pi * np.random.rand()) +
        max_deformation * np.sin(random_numbers[1] * angles + 2 * np.pi * np.random.rand()) +
        max_deformation * np.sin(random_numbers[2] * angles + 2 * np.pi * np.random.rand()) +
        max_deformation * np.sin(random_numbers[3] * angles + 2 * np.pi * np.random.rand()) +
        max_deformation * np.sin(random_numbers[4] * angles + 2 * np.pi * np.random.rand())
    )
    radius = base_radius + deformation
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return x, y

def stretch_and_rotate(x, y):
    scale_x = np.random.uniform(0.5, 3)
    scale_y = np.random.uniform(0.5, 3)
    if np.random.rand() > 0.5:
        x *= scale_x
    else:
        y *= scale_y
    theta = np.random.uniform(0, np.pi / 2)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_rot = x * cos_theta - y * sin_theta
    y_rot = x * sin_theta + y * cos_theta
    return x_rot, y_rot

def make_stain(image_size=50):
    image = np.ones((image_size, image_size))
    x, y = generate_random_blob()
    x, y = stretch_and_rotate(x, y)
    x = ((x + 3) / 6 * image_size).astype(int)
    y = ((y + 3) / 6 * image_size).astype(int)
    x = np.clip(x, 0, image_size - 1)
    y = np.clip(y, 0, image_size - 1)
    rr, cc = polygon(y, x)
    image[rr, cc] = 0
    return image

def locate_stain(drop, image, image_size):
    intensity = int(random.uniform(1, 100)) - 1
    image = image * intensity
    drop = drop.astype("int")
    image[image == 0], image[image == intensity] = intensity + 1, 0
    image_size_half = int(image_size / 2)
    y = int(random.uniform(image_size_half + 1, drop.shape[0] - image_size_half - 1))
    x = int(random.uniform(image_size_half + 1, drop.shape[1] - image_size_half - 1))
    region = drop[y - image_size_half:y + image_size_half, x - image_size_half:x + image_size_half, :]
    image_3d = np.stack((image,) * 3, axis=-1)
    condition = region <= 220
    region[condition] = region[condition] - image_3d[condition]
    drop[y - image_size_half:y + image_size_half, x - image_size_half:x + image_size_half, :] = region
    drop[drop < 0] = 0
    drop = 255 - drop
    return drop.astype("uint8")

def generate_circular_gradient(size=36):
    center = size // 2
    y, x = np.ogrid[:size, :size]
    distance = np.sqrt((x - center)**2 + (y - center)**2)
    max_distance = np.sqrt(2) * center 
    gradient = distance / max_distance + 0.01
    return 1 - gradient

def extract_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def create_circular_gradient(radius):
    diameter = 2 * radius
    gradient = np.zeros((diameter, diameter), dtype=np.float32)
    for y in range(diameter):
        for x in range(diameter):
            distance = np.sqrt((x - radius) ** 2 + (y - radius) ** 2)
            if distance <= radius:
                gradient[y, x] = 1 - (distance / radius)
    return gradient

def apply_gradient_to_image(image, position, radius):
    gradient = create_circular_gradient(radius)
    overlay = np.zeros_like(image, dtype=np.float32)
    x, y = position
    h, w = image.shape[:2]
    x_start = max(x - radius, 0)
    y_start = max(y - radius, 0)
    x_end = min(x + radius, w)
    y_end = min(y + radius, h)
    overlay[y_start:y_end, x_start:x_end, :] = cv2.merge(
        [gradient[y_start - (y - radius):y_end - (y - radius),
                  x_start - (x - radius):x_end - (x - radius)]] * 3)
    image = image.astype(np.float32) / 255.0
    combined = image + overlay
    combined = np.clip(combined, 0, 1)
    return (combined * 255).astype(np.uint8)

def glow_spot(image, max_radius):
    image = 255 - image
    image = image.astype("uint8")
    edges = extract_edges(image)
    edge_coords = np.column_stack(np.where(edges > 0))
    selected_edge = random.choice(edge_coords)
    radius = int(random.uniform(1, max_radius))
    return apply_gradient_to_image(image, (selected_edge[1], selected_edge[0]), radius)

