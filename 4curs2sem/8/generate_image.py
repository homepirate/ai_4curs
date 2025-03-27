import random
from PIL import Image, ImageDraw

def get_bbox(shape, center, circle_radius, square_side, padding=1):
    x, y = center
    if shape == "circle":
        r = circle_radius
        return (x - r - padding, y - r - padding, x + r + padding, y + r + padding)
    else:  # square
        half = square_side // 2
        return (x - half - padding, y - half - padding, x + half + padding, y + half + padding)

def bbox_intersect(b1, b2):
    return not (b1[2] <= b2[0] or b1[0] >= b2[2] or b1[3] <= b2[1] or b1[1] >= b2[3])

def generate_random_shapes(image_size, circle_radius, square_side,
                           min_circles=3, max_circles=8,
                           min_squares=3, max_squares=8,
                           max_attempts=1000):
    width, height = image_size
    shapes = []

    def place(shape, count):
        for _ in range(count):
            for _ in range(max_attempts):
                if shape == "circle":
                    x = random.randint(circle_radius + 1, width - circle_radius - 1)
                    y = random.randint(circle_radius + 1, height - circle_radius - 1)
                else:
                    half = square_side // 2
                    x = random.randint(half + 1, width - half - 1)
                    y = random.randint(half + 1, height - half - 1)
                bbox = get_bbox(shape, (x, y), circle_radius, square_side)
                if not any(bbox_intersect(bbox, get_bbox(s, c, circle_radius, square_side)) for s, c in shapes):
                    shapes.append((shape, (x, y)))
                    break

    place("circle", random.randint(min_circles, max_circles))
    place("square", random.randint(min_squares, max_squares))
    return shapes

def create_image(image_size, shapes, circle_radius, square_side, shape_color):
    img = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(img)
    for shape, (x, y) in shapes:
        if shape == "circle":
            draw.ellipse((x-circle_radius, y-circle_radius, x+circle_radius, y+circle_radius),
                         fill=shape_color)
        else:
            half = square_side // 2
            draw.rectangle((x-half, y-half, x+half, y+half), fill=shape_color)
    return img

if __name__ == "__main__":
    IMAGE_SIZE = (400, 400)
    CIRCLE_RADIUS = 10
    SQUARE_SIDE = 20

    # Generate and save black‑white test image
    bw_shapes = generate_random_shapes(IMAGE_SIZE, CIRCLE_RADIUS, SQUARE_SIDE)
    bw_img = create_image(IMAGE_SIZE, bw_shapes, CIRCLE_RADIUS, SQUARE_SIDE, "black")
    bw_img.save("random_test_bw.png")

    # Generate and save color test image (red shapes)
    color_shapes = generate_random_shapes(IMAGE_SIZE, CIRCLE_RADIUS, SQUARE_SIDE)
    color_img = create_image(IMAGE_SIZE, color_shapes, CIRCLE_RADIUS, SQUARE_SIDE, (255, 0, 0))
    color_img.save("random_test_color.png")

    print(f"Generated {len([s for s,_ in bw_shapes if s=='circle'])} circles and "
          f"{len([s for s,_ in bw_shapes if s=='square'])} squares in BW image.")
    print(f"Generated {len([s for s,_ in color_shapes if s=='circle'])} circles and "
          f"{len([s for s,_ in color_shapes if s=='square'])} squares in color image.")
