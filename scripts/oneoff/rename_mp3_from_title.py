import os
import eyed3

# Function to rename the MP3 file based on its title
def rename_mp3_with_title(mp3_file):
    audiofile = eyed3.load(mp3_file)
    if audiofile.tag and audiofile.tag.title:
        new_filename = f"{audiofile.tag.title}.mp3"
        new_filename = new_filename.replace('/', '_').replace(': ', '')  # Replace any slashes in the title with underscores
        new_path = os.path.join(os.path.dirname(mp3_file), new_filename)
        try:
            os.rename(mp3_file, new_path)
            print(f"Renamed {mp3_file} to {new_path}")
        except:
            return

# Directory path containing the MP3 files
folder_path = "."

# Loop through all files in the directory
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.lower().endswith(".mp3"):
            mp3_file = os.path.join(root, file)
            rename_mp3_with_title(mp3_file)

