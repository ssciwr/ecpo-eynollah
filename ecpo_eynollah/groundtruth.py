"""Massaging of existing ground truth into desired formats """

import re
import click
import functools
import io
import json
import pathlib
import requests
import svgelements
import tqdm

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
    url = re.sub(r"http://localhost:[0-9]*/", "", url)

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
        # Create the config for this annotation
        #

        config = {
            "original_width": width,
            "original_height": height,
            "image_rotation": 0,
            "id": target["id"],
            "from_name": "label",
            "to_name": "image",
        }

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

            config["value"] = {
                "points": points,
                "closed": True,
                "labels": [label],
            }
            config["type"] = "polygonlabels"

            yield config
            return

        if isinstance(svg, svgelements.Circle):
            config["value"] = {
                "x": svg.cx / width * 100.0,
                "y": svg.cy / height * 100.0,
                "radiusX": svg.rx / width * 100.0,
                "radiusY": svg.ry / height * 100.0,
                "rotation": 0,
                "labels": [label],
            }
            config["type"] = "ellipse"

            yield config
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


def ecpo_data_to_labelstudio(
    input, output, modify, cors_anywhere_port=None, mark_as_annotations=False
):
    # Find all valid JSON files in the input
    json_files = list(filter(_is_data_json, pathlib.Path(input).rglob("*.json")))

    # The resulting - very large - structure
    tasks = []

    # Iterate over the JSONs
    for i, filename in tqdm.tqdm(enumerate(json_files)):
        with open(filename, "r") as f:
            data = json.load(f)

        # Get all annotations from this JSON file
        annotations = image_annotations_to_labelstudio(data)

        # Maybe modify the annotations:
        if modify:
            annotations = modify_annotations_for_eynollah(annotations)

        cors_proxy = ""
        if cors_anywhere_port is not None:
            cors_proxy = f"http://localhost:{cors_anywhere_port}/"

        # Create the required IIIF URL
        iiif = (
            cors_proxy
            + iiif_metadata(data["items"][0]["target"][0]["source"])["id"]
            + "/full/full/0/default.jpg"
        )

        key = "annotations" if mark_as_annotations else "predictions"

        tasks.append(
            {
                "id": i,
                "data": {"image": iiif, "name": filename.stem},
                key: [{"result": annotations}],
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

    for task in tqdm.tqdm(data):
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

                # This is a polygon
                if "points" in annotation:
                    draw.polygon(
                        [_percentage_to_pixels(c) for c in annotation["points"]],
                        fill=colormap[annotation["labels"][0]],
                    )

                # This is an ellipse
                if "radiusX" in annotation:
                    center = _percentage_to_pixels((annotation["x"], annotation["y"]))
                    radius = _percentage_to_pixels(
                        (annotation["radiusX"], annotation["radiusY"])
                    )
                    bbox = [
                        center[0] - radius[0],
                        center[1] - radius[1],
                        center[0] + radius[0],
                        center[1] + radius[1],
                    ]
                    draw.ellipse(bbox, fill=colormap[annotation["labels"][0]])

                # This is a rectangle
                if "width" in annotation:
                    center = _percentage_to_pixels((annotation["x"], annotation["y"]))
                    size = _percentage_to_pixels(
                        (annotation["width"], annotation["height"])
                    )
                    bbox = [
                        center[0] - size[0] / 2,
                        center[1] - size[1] / 2,
                        center[0] + size[0] / 2,
                        center[1] + size[1] / 2,
                    ]
                    draw.rectangle(bbox, fill=colormap[annotation["labels"][0]])

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
@click.option(
    "--cors-anywhere-port",
    type=int,
    default=None,
    help="The port of a local cors-anywhere server to prefix IIIF URLs with",
)
@click.option(
    "--mark-as-annotations/--no-mark-as-annotations",
    type=bool,
    default=False,
    help="Whether to mark the output as annotations instead of predictions (LabelStudio terminology).",
)
@click.command
def ecpo_data_to_labelstudio_cli(
    input, output, modify, cors_anywhere_port, mark_as_annotations
):
    ecpo_data_to_labelstudio(
        input, output, modify, cors_anywhere_port, mark_as_annotations
    )


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
