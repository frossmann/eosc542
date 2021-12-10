#%%
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as pltw


filename = "/Users/francis/repos/volc/mars_glaciovolcanic_edifices_filt.csv"
df = pd.read_csv(filename)

# can you cross reference this with the 2003 paper?
# %%
lon = df["lon"]
lat = df["lat"]
height = df["height"]

fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.hist(lat)
ax2.hist(lon)
ax3.hist(height)
# %%
from PIL import Image

img_name = "/Users/francis/Nextcloud/Mars_MGS_MOLA_DEM_mosaic_global_463m.tiff"
im = Image.open(img_name)
im.show()

#%%
import numpy as np

a = np.array([1, 2])
# %%
