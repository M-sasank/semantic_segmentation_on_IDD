import os
import re

def rename_files(directory):
    for filename in os.listdir(directory):
        if "leftImg8bit" in filename:
            os.rename(os.path.join(directory, filename), os.path.join(directory, filename.replace("leftImg8bit", "gtFine_labelTrainIds")))

# Example usage:
directory = "/media/ram/338f6363-03b7-4ad7-a2be-40c31f59dee4/20230418_backup/ram/Students/B.Tech_2020/sasank/InternImage/segmentation/data/cityscapes/gtFine/train"
rename_files(directory)
