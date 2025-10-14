"""Massaging of existing ground truth into desired formats """

import click
import io
import json
import os
import pathlib
import requests
import svgelements
import tifffile

from urllib.parse import unquote


def _recursive_unquote(url):
    """Unquote a URL recursively until it does not change anymore"""

    unquoted = unquote(url)
    if url == unquoted:
        return url

    return unquote(unquoted)


def _find_filename_with_inconsistent_naming_scheme(filename):
    _variants = [lambda fn: fn, lambda fn: fn.replace("+", "to")]

    for var in _variants:
        candidate = var(filename)
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"The file {filename} was nowhere to be found")


def _is_data_json(filename):
    """Some rules to exclude JSON files in globbing that occurred during development"""
    if ".ipynb_checkpoints" in str(filename):
        return False
    if str(filename).endswith("generated-schema.json"):
        return False

    return True


def annotation_to_labelstudio(annotation, image_root):
    """Translates a single annotation from ecpo-annotate to LabelStudio"""

    def _body_target_combination(body, target):
        # To my understanding, we can omit GroupAnnotation. These seem to occur when
        # there are multiple bodies and two targets - one being a CategoryLabel and the
        # other being a GroupAnnotation. I think that this is used to denote which text
        # parts form an article. For LabelStudio, we are not actually interested in this
        # information.
        if body["type"] == "GroupAnnotation":
            return

        #
        # Input sanitizing: Any exception here means we have a new TODO
        #

        if body["type"] != "CategoryLabel":
            raise ValueError(f"Got unknown label type: {body['type']}")
        if target["type"] != "SpecificResource":
            raise ValueError(
                f"Got target type other than 'SpecificResource': {target['type']}"
            )
        if target["selector"]["type"] != "SvgSelector":
            raise ValueError(
                f"Got selector other than 'SvgSelector': {target['selector']['type']}"
            )

        #
        # Extract some required data
        #

        label = body["value"]["name"]
        selector = target["selector"]["value"]

        #
        # Find image data for this annotation (wild)
        #

        image_url = target["source"]

        # We have double-encoded URLs
        image_url = _recursive_unquote(image_url)

        # We need to remove all the IIIF stuff - we want to use local data,
        # otherwise we are creating a lot of network traffic to get data that
        # we already have readily available
        filename = image_url.replace(
            "https://kjc-sv002.kjc.uni-heidelberg.de:8080/fcgi-bin/iipsrv.fcgi?IIIF=imageStorage/ecpo_new/",
            "",
        ).replace("/full/full/0/default.jpg", "")
        filename = _find_filename_with_inconsistent_naming_scheme(
            str(image_root / filename)
        )

        # Extract the image size
        with tifffile.TiffFile(filename) as tif:
            # Each level of the pyramid is a "page"
            original_sized = tif.pages[0]
            width = original_sized.imagewidth
            height = original_sized.imagelength

        #
        # Convert the SVG selector to global coordinates
        #

        svg = svgelements.SVG.parse(io.StringIO(selector))
        if len(svg) > 1:
            raise ValueError("Expected only one element in SVG string")
        svg = svg[0]
        if not isinstance(svg, svgelements.Polygon):
            print(f"Omitting SVG element of type {type(svg)}")
            return

        # Transform into simple data structure for JSON dump. LabelStudio uses
        # a percentage of the original size as data representation.
        points = [[p.x / width * 100.0, p.y / height * 100.0] for p in svg]

        yield {
            "original_width": width,
            "original_height": height,
            "image_rotation": 0,
            "value": {
                "points": points,
                "closed": True,
                "polygonlabels": [label],
            },
            "id": target["id"],
            "from_name": "tag",
            "to_name": "img",
            "type": "polygonlabels",
            "origin": "ecpo-data",
        }

    # Traverse the combinations of body/target
    for body in annotation["body"]:
        for target in annotation["target"]:
            yield from _body_target_combination(body, target)


def image_annotations_to_labelstudio(data, image_root):
    """Translates all annotations on a single image to LabelStudio"""

    annotations = []
    for annot in data["items"]:
        for polygon in annotation_to_labelstudio(annot, image_root):
            annotations.append(polygon)

    return annotations


def ecpo_data_to_labelstudio(input, output, image_root):
    # Find all valid JSON files in the input
    json_files = list(filter(_is_data_json, pathlib.Path(input).rglob("*.json")))

    # The resulting - very large - structure
    tasks = []

    # Iterate over the JSONs
    for i, filename in enumerate(json_files):
        with open(filename, "r") as f:
            data = json.load(f)

        # Get all annotations from this JSON file
        annotations = image_annotations_to_labelstudio(data, pathlib.Path(image_root))

        # Create a live URL for use with LabelStudio without duplicating the
        # data on our LabelStudio VM.
        old_iiif = data["items"][0]["target"][0]["source"]
        old_iiif = _recursive_unquote(old_iiif)
        iiif = old_iiif.replace(
            "https://kjc-sv002.kjc.uni-heidelberg.de:8080/fcgi-bin/iipsrv.fcgi?IIIF=imageStorage/ecpo_new/",
            "https://ecpo.cats.uni-heidelberg.de/fcgi-bin/iipsrv.fcgi?IIIF=/ecpo/images/",
        )
        iiif = iiif.replace("+", "to")

        # Check the validity of the IIIF URL by accessing the metadata file
        info_url = iiif.replace("/full/full/0/default.jpg", "/info.json")
        r = requests.head(info_url, timeout=5)
        if r.status_code != 200:
            raise ValueError(f"This IIIF URL seems to be incorrect: {iiif}")

        tasks.append(
            {"id": i, "data": {"image": iiif}, "annotations": [{"result": annotations}]}
        )

    with open(pathlib.Path(output), "w") as f:
        json.dump(tasks, f)


@click.option(
    "--input",
    type=click.Path(
        writable=False,
        file_okay=False,
        dir_okay=True,
        exists=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    default="./data/annotator_data/JB-visualGT",
    help="The location of the ground truth input files",
)
@click.option(
    "--output",
    type=click.Path(
        writable=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    default="labelstudio.json",
    help="Where to write the LabelStudio output. Must be a directory.",
)
@click.option(
    "--image-root",
    type=click.Path(
        writable=False,
        file_okay=False,
        dir_okay=True,
        exists=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    help="The root folder of the (local) image directory e.g. your SDS@HD mount",
)
@click.command
def cli(input, output, image_root):
    ecpo_data_to_labelstudio(input, output, image_root)


if __name__ == "__main__":
    ecpo_data_to_labelstudio(
        pathlib.Path("../ecpo-data/data/annotator_data/JB-visualGT/"),
        pathlib.Path("labelstudio_data"),
        pathlib.Path("/home/dkempf/heibox/ECPO_data/images_with_groundtruth/ptif"),
    )
