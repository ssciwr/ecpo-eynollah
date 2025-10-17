"""Massaging of existing ground truth into desired formats """

import click
import functools
import io
import json
import pathlib
import requests
import svgelements

from PIL import Image, ImageDraw
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
    # If this contains a cors-anywhere prefix, we omit it, because these URLs
    # *only* work from inside the browser.
    url = url.replace("http://localhost:7000/", "")

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


def modify_annotations_for_eynollah(annotations):
    """Apply our modifications to the ecpo_data"""

    # Analyse duplications of labels
    id_to_labels = {}
    for annot in annotations:
        id_to_labels.setdefault(annot["id"], [])
        id_to_labels[annot["id"]].append(annot["value"]["labels"][0])

    # Whitelist the annotations that we want to keep
    result = []
    for annot in annotations:
        # This was already handled!
        if len(id_to_labels[annot["id"]]) == 0:
            continue

        # We have more than one label!
        if len(id_to_labels[annot["id"]]) > 1:
            # Is it the same label? We remove one label from
            # our dictionary and continue. The next occurence
            # will add it to the result
            if len(set(id_to_labels[annot["id"]])) == 1:
                id_to_labels[annot["id"]].remove(annot["value"]["labels"][0])
                continue

            # Treat special case: image + article -> heading
            if (
                "image" in id_to_labels[annot["id"]]
                and "article" in id_to_labels[annot["id"]]
            ):
                annot["value"]["labels"][0] = "heading"
                id_to_labels[annot["id"]].clear()
                result.append(annot)
                continue

            if "additional" in id_to_labels[annot["id"]]:
                annot["value"]["labels"][0] = "text"
                id_to_labels[annot["id"]].clear()
                result.append(annot)
                continue

            if (
                "image" in id_to_labels[annot["id"]]
                and "advertisement" in id_to_labels[annot["id"]]
            ):
                annot["value"]["labels"][0] = "image"
                id_to_labels[annot["id"]].clear()
                result.append(annot)
                continue

            if (
                "article" in id_to_labels[annot["id"]]
                and "advertisement" in id_to_labels[annot["id"]]
            ):
                annot["value"]["labels"][0] = "article"
                id_to_labels[annot["id"]].clear()
                result.append(annot)
                continue

            print(f"Unhandled pair of two labels: {id_to_labels[annot['id']]}")

        # Rename labels
        if annot["value"]["labels"][0] == "article":
            annot["value"]["labels"][0] = "text"
        if annot["value"]["labels"][0] == "additional":
            annot["value"]["labels"][0] = "text"
        if annot["value"]["labels"][0] == "advertisement":
            annot["value"]["labels"][0] = "text"

        result.append(annot)

    return result


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
                    "labels": [label],
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


def ecpo_data_to_labelstudio(input, output, modify):
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

        # Maybe modify the annotations:
        if modify:
            annotations = modify_annotations_for_eynollah(annotations)

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
                "predictions": [{"result": annotations}],
            }
        )

    with open(pathlib.Path(output), "w") as f:
        json.dump(tasks, f)


def labelstudio_to_png(input, output, color):
    """Create PNGs from LabelStudio annotations"""

    # Define the label priority from low to high. This is used to resolve
    # overlapping annotations gracefully.
    label_priority = [
        "text",
        "image",
        "heading",
        "separator",
    ]

    mode = "RGB" if color else "L"
    background = (0, 0, 0) if color else 0
    colormap = {
        "text": (231, 76, 60) if color else 1,
        "image": (52, 152, 219) if color else 2,
        "heading": (230, 126, 34) if color else 3,
        "separator": (155, 89, 182) if color else 4,
    }

    # Read the exported data
    with open(input, "r") as f:
        data = json.load(f)

    for task in data:
        # Get the image size and instantiate an empty image
        metadata = iiif_metadata(task["image"])
        image = Image.new(mode, (metadata["width"], metadata["height"]), background)

        # Create a drawing context
        draw = ImageDraw.Draw(image)

        def _percentage_to_pixels(coord):
            return (
                coord[0] / 100 * metadata["width"],
                coord[1] / 100 * metadata["height"],
            )

        for label in label_priority:
            for annotation in task["label"]:
                if annotation["labels"][0] != label:
                    continue

                draw.polygon(
                    [_percentage_to_pixels(c) for c in annotation["points"]],
                    fill=colormap[annotation["labels"][0]],
                )

        # Create output path
        filename = output / f"{task['name']}.png"
        image.save(filename, "PNG")


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
    default="labelstudio/labelstudio.json",
    help="Where to write the LabelStudio output. Must be a directory.",
)
@click.option(
    "--modify/--no-modify",
    type=bool,
    default=False,
    help="Whether to modify annotations for eynollah",
)
@click.command
def ecpo_data_to_labelstudio_cli(input, output, modify):
    ecpo_data_to_labelstudio(input, output, modify)


@click.option(
    "--input",
    type=click.Path(
        writable=False,
        file_okay=True,
        dir_okay=False,
        exists=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    help="The JSON-MIN export from LabelStudio",
)
@click.option(
    "--output",
    type=click.Path(
        writable=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    default="labelstudio/png_output",
    help="The directory where PNG output is placed",
)
@click.option(
    "--color/--no-color",
    type=bool,
    default=False,
    help="Whether to make this colorful (mainly for debugging and visualization)",
)
@click.command
def labelstudio_to_png_cli(input, output, color):
    output.mkdir(exist_ok=True, parents=True)
    labelstudio_to_png(input, output, color)


if __name__ == "__main__":
    ecpo_data_to_labelstudio(
        pathlib.Path("../ecpo-data/data/annotator_data/JB-visualGT/"),
        pathlib.Path("labelstudio/labelstudio_data.json"),
        True,
    )
