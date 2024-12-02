# rename all files in folder `instrumentals` (which have names of the form `{integer}_instrumental.mp3`) to `{integer}.mp3`
# same for `vocals` folder

# code:

import os
import re

def rename_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3"):
            new_filename = re.sub(r"_instrumental", "", filename)
            os.rename(f"{folder_path}/{filename}", f"{folder_path}/{new_filename}")

rename_files("instrumentals")
rename_files("vocals")