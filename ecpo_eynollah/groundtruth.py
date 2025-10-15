"""Massaging of existing ground truth into desired formats """

import click
import functools
import io
import json
import pathlib
import requests
import svgelements

from urllib.parse import unquote


def _recursive_unquote(url):
    """Unquote a URL recursively until it does not change anymore"""

    unquoted = unquote(url)
    if url == unquoted:
        return url

    return unquote(unquoted)


def _is_data_json(filename):
    """Some rules to exclude JSON files in globbing that occurred during development"""
    if ".ipynb_checkpoints" in str(filename):
        return False
    if str(filename).endswith("generated-schema.json"):
        return False

    return True


@functools.lru_cache
def iiif_metadata(url):
    """Access IIIF metadata for the given URL. Also corrects the URL from
    old server in the ecpo-data repository to the currently running one.
    """
    # Create a live URL for use with LabelStudio without duplicating the
    # data on our LabelStudio VM.
    old_iiif = _recursive_unquote(url)
    iiif = old_iiif.replace(
        "https://kjc-sv002.kjc.uni-heidelberg.de:8080/fcgi-bin/iipsrv.fcgi?IIIF=imageStorage/ecpo_new/",
        "https://ecpo.cats.uni-heidelberg.de/fcgi-bin/iipsrv.fcgi?IIIF=/ecpo/images/",
    )
    iiif = iiif.replace("+", "to")

    # Check the validity of the IIIF URL by accessing the metadata file
    info_url = iiif.replace("/full/full/0/default.jpg", "/info.json")
    return requests.get(info_url, timeout=5).json()


def annotation_to_labelstudio(annotation):
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

        metadata = iiif_metadata(target["source"])
        width = metadata["width"]
        height = metadata["height"]

        #
        # Convert the SVG selector to global coordinates
        #

        svg = svgelements.SVG.parse(io.StringIO(selector))
        if len(svg) > 1:
            raise ValueError("Expected only one element in SVG string")
        svg = svg[0]

        # Treat polygonal annotations
        if isinstance(svg, svgelements.Polygon):
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
                "from_name": "label",
                "to_name": "image",
                "type": "polygonlabels",
            }
            return

        if isinstance(svg, svgelements.Circle):
            print("Circle found")
            return

        print(f"Omitting SVG element of type {type(svg)}")

    # Traverse the combinations of body/target
    for body in annotation["body"]:
        for target in annotation["target"]:
            yield from _body_target_combination(body, target)


def image_annotations_to_labelstudio(data):
    """Translates all annotations on a single image to LabelStudio"""

    annotations = []
    for annot in data["items"]:
        for polygon in annotation_to_labelstudio(annot):
            annotations.append(polygon)

    return annotations


def ecpo_data_to_labelstudio(input, output):
    # Find all valid JSON files in the input
    json_files = list(filter(_is_data_json, pathlib.Path(input).rglob("*.json")))

    # The resulting - very large - structure
    tasks = []

    # Iterate over the JSONs
    for i, filename in enumerate(json_files):
        with open(filename, "r") as f:
            data = json.load(f)

        # Get all annotations from this JSON file
        annotations = image_annotations_to_labelstudio(data)

        # Create the required IIIF URL
        iiif = (
            "http://localhost:7000/"
            + iiif_metadata(data["items"][0]["target"][0]["source"])["id"]
            + "/full/full/0/default.jpg"
        )

        tasks.append(
            {
                "id": i,
                "data": {"image": iiif, "name": filename.stem},
                "annotations": [{"result": annotations}],
            }
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
@click.command
def cli(input, output):
    ecpo_data_to_labelstudio(input, output)


if __name__ == "__main__":
    ecpo_data_to_labelstudio(
        pathlib.Path("../ecpo-data/data/annotator_data/JB-visualGT/"),
        pathlib.Path("labelstudio_data.json"),
    )
