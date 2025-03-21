from PIL import Image, ImageDraw, ImageFont

def create_test_image(text, filename):
    img = Image.new('RGB', (300, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), text, fill=(0, 0, 0))
    img.save(filename)


# Use this format to create test images with different food-related phrases
create_test_image("Golden pancakes stack.", "test33.png")
create_test_image("Savory tomato basil soup.", "test34.png")
