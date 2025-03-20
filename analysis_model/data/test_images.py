from PIL import Image, ImageDraw, ImageFont

def create_test_image(text, filename):
    img = Image.new('RGB', (300, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), text, fill=(0, 0, 0))
    img.save(filename)

create_test_image("Free Pizza at 6 PM!", "test1.png")
create_test_image("Join us for dinner!", "test2.png")