from PIL import Image
from PIL import ImageChops
import sys

x = Image.open(sys.argv[1])
xpixels = x.load()
width, height = x.size
y = Image.open(sys.argv[2])
ypixels = y.load()
for i in range(width):
    for j in range(height):
        xpixel = xpixels[i, j]
        ypixel = ypixels[i, j]
        if( xpixel == ypixel ):
            ypixels[i, j] = 0,0,0,0
y.save("ans_two.png")
