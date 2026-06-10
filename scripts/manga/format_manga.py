import json
import os
import time

import piexif
import requests

import manga.local_config as config
from Rignak.init import assert_argument_types, ExistingFilename, logger
from Rignak.path import listdir


@assert_argument_types
def get_metadata(
        folder: ExistingFilename
) -> dict:
    json_filename = os.path.join(folder, "info.json")
    with open(json_filename, 'r') as file:
        metadata = json.load(file).get('manga_info', {})
    return metadata


@assert_argument_types
def update_tags(
        folder: ExistingFilename, metadata: dict
) -> None:
    tags = metadata.get('characters', []) + metadata.get('tags', {}).get('misc', [])
    tags = ', '.join(tags).encode("utf-16le")

    for filename in listdir(folder, extensions=('.jpg', '.png')):
        exif_dict = piexif.load(filename)
        exif_dict["0th"][piexif.ImageIFD.XPKeywords] = tags
        piexif.insert(piexif.dump(exif_dict), filename)


@assert_argument_types
def update_convention_name(
        folder: ExistingFilename,
        metadata: dict,
        api_url: str = config.API_URL,
        cooldown: int = config.COOLDOWN
) -> None:
    url = metadata.get('url')[:-1]
    url, suffix = os.path.split(url)
    url, prefix = os.path.split(url)

    payload = {"method": "gdata", "gidlist": [[int(prefix), suffix]], "namespace": 1}
    metadata = json.loads(requests.post(api_url, json=payload).text)
    time.sleep(cooldown)

    galleries = metadata.get('gmetadata', [])
    if galleries:
        convention = galleries[0].get('title', '').split()[0]
        convention = f"[{convention[1:-1]}]"
        new_folder = folder.replace('[Unknown Convention]', convention)
        os.rename(folder, new_folder)


@assert_argument_types
def format_folder(
        folder: ExistingFilename
) -> None:
    metadata = get_metadata(folder)
    update_tags(folder, metadata)
    update_convention_name(folder, metadata)


@assert_argument_types
def main(
        root: ExistingFilename = config.TODO_ROOT
) -> None:
    folders = listdir(root)

    logger.set_iterator(len(folders))
    for folder in folders:
        folder = ExistingFilename(folder)
        format_folder(folder)

        logger.iterate(os.path.basename(folder))
        break


if __name__ == "__main__":
    main()
