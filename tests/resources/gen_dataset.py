import numpy as np
import PIL
from glob import glob
import config

Q = sorted(glob(config.root_dir + "/tests/resources/test_query_img/*"))
M = sorted(glob(config.root_dir + "/tests/resources/test_db_img/*"))[:30]
print(M)

#name = 1
#for i in range(30):
#   img = np.random.rand(320, 320, 3) * 255
#    img = PIL.Image.fromarray(img.astype(np.uint8))
#    img.save("test_db_img/" + str(name) + ".png")
#    name += 1
