"""Massaging of existing ground truth into desired formats"""

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

from shapely.geometry import Polygon

import math


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


def labelstudio_to_png(
    input, output, color, overwrite, artboundary_buffer_size=10, considered_ids=None
):
    """Create PNGs from LabelStudio annotations"""

    # special label used to draw boundaries around other annotations
    artificial_boundary = "artBoundary"

    # Define the label priority from low to high. This is used to resolve
    # overlapping annotations gracefully.
    label_priority = [
        artificial_boundary,
        "text",
        "advertisement",
        "image",
        "heading",
        "separator",
        "additional",
        "article",
    ]  # advertisement, additional, article are not used in LS output.

    mode = "RGB" if color else "L"
    background = (0, 0, 0) if color else 0
    colormap = {
        artificial_boundary: (255, 255, 255) if color else 1,
        "text": (231, 76, 60) if color else 2,
        "article": (231, 76, 60) if color else 2,
        "image": (52, 152, 219) if color else 3,
        "heading": (230, 126, 34) if color else 4,
        "separator": (
            (155, 89, 182) if color else 5
        ),  # we only use up to here, 6 classes in total
        "advertisement": (46, 204, 113) if color else 6,
        "additional": (52, 73, 94) if color else 7,
    }

    # Read the exported data
    with open(input, "r") as f:
        data = json.load(f)

    print(
        f"Creating PNGs with {artboundary_buffer_size} pixels buffer size."
        f"Considering IDs: {considered_ids}"
    )

    # record image with rotated rectangles
    num_rotated_rectangles = 0
    ro_rec_files = []

    for task in tqdm.tqdm(data):
        # only consider task with the given index for debugging
        if considered_ids is not None and task["id"] not in considered_ids:
            continue

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

        def _get_buffer_points(annotation, buffer_size, annotation_type="polygon"):
            """Get the points of a buffered annotation."""
            if annotation_type == "polygon":
                org_points = [_percentage_to_pixels(c) for c in annotation["points"]]
                if buffer_size == 0:
                    return org_points

                org_polygon = Polygon(org_points)
                buffered_polygon = org_polygon.buffer(buffer_size, join_style="mitre")
                return list(buffered_polygon.exterior.coords)
            elif annotation_type == "ellipse":
                cx, cy = _percentage_to_pixels((annotation["x"], annotation["y"]))
                rx, ry = _percentage_to_pixels(
                    (annotation["radiusX"], annotation["radiusY"])
                )
                if buffer_size == 0:
                    return [cx, cy, rx, ry]

                return [cx, cy, rx + buffer_size, ry + buffer_size]
            else:
                # reactangle
                top_left = _percentage_to_pixels((annotation["x"], annotation["y"]))
                size = _percentage_to_pixels(
                    (annotation["width"], annotation["height"])
                )

                if buffer_size == 0:
                    return [top_left[0], top_left[1], size[0], size[1]]

                return [
                    top_left[0] - buffer_size,
                    top_left[1] - buffer_size,
                    size[0] + 2 * buffer_size,
                    size[1] + 2 * buffer_size,
                ]

        def _get_buffer_info(annotation, buffer_size, annotation_type="polygon"):
            """Get buffered points and the corresponding filling color"""
            points_info = _get_buffer_points(annotation, buffer_size, annotation_type)
            fill_color = colormap[annotation["labels"][0]]

            if buffer_size > 0:  # artifical boundary
                return points_info, colormap[artificial_boundary]

            return points_info, fill_color

        def _draw_annotation(annotation, buffer_size=10):
            """Draw an annotation on the image.
            The buffer_size is used to draw an additional boundary around the real annotation.
            """

            if buffer_size < 0:
                raise ValueError("Buffer size must be non-negative")

            # This is a polygon
            if "points" in annotation:
                # covert points to polygon
                target_points, fill_color = _get_buffer_info(
                    annotation, buffer_size, annotation_type="polygon"
                )
                draw.polygon(
                    target_points,
                    fill=fill_color,
                )

            # This is an ellipse
            if "radiusX" in annotation:
                target_points, fill_color = _get_buffer_info(
                    annotation, buffer_size, annotation_type="ellipse"
                )
                cx, cy, rx, ry = target_points
                bbox = [
                    cx - rx,
                    cy - ry,
                    cx + rx,
                    cy + ry,
                ]
                draw.ellipse(bbox, fill=fill_color)

            # This is a rectangle
            if "width" in annotation:
                target_points, fill_color = _get_buffer_info(
                    annotation, buffer_size, annotation_type="rectangle"
                )
                x, y, width, height = target_points

                # consider rotation if available!
                if "rotation" in annotation and annotation["rotation"] != 0:
                    # draw the rectangle differently
                    # ATTENTION! labelstudio stores rotation around top-left cornor,
                    # not around the center

                    # corners relative to top-left corner
                    # note that y-axis is downward in PIL
                    # if y-axis is upward, the corners should be:
                    # (0, 0), (width, 0), (width, -height), (0, -height)
                    corners = [
                        (0, 0),  # top-left corner
                        (width, 0),  # top-right corner
                        (width, height),  # bottom-right corner
                        (0, height),  # bottom-left corner
                    ]

                    # labelstudio stores rotation in range 0-360,
                    # here, we keep rotation degree as is
                    angle_rad = math.radians(annotation["rotation"])

                    rotated_corners = []
                    for dx, dy in corners:
                        # apply rotation
                        # in case of upward y-axis, rotate at top-left corner,
                        # the formular will be:
                        # rx = dx * cos(angle_rad) + dy * math.sin(angle_rad)
                        # ry = - dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
                        # after considering the y-axis direction, it becomes:
                        rx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
                        ry = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
                        rotated_corners.append((x + rx, y + ry))

                    draw.polygon(rotated_corners, fill=fill_color)
                else:
                    bbox = [
                        x,
                        y,
                        x + width,
                        y + height,
                    ]
                    draw.rectangle(bbox, fill=fill_color)

        for label in label_priority:
            for annotation in task["label"]:
                # record rotated rectangles for debugging
                if "rotation" in annotation and annotation["rotation"] != 0:
                    # # debug
                    # print(
                    #     f"Found rotated rectangle in file {task['name']}. Rotation angle: {annotation['rotation']} degrees."
                    # )
                    num_rotated_rectangles += 1
                    ro_rec_files.append(f"{task["name"]}-{annotation["labels"][0]}")
                # draw artifical boundaries first
                if label == artificial_boundary:
                    _draw_annotation(
                        annotation, buffer_size=artboundary_buffer_size
                    )  # buffer for artificial boundaries
                else:
                    # draw other annotations in order of priority

                    if annotation["labels"][0] != label:
                        continue

                    _draw_annotation(
                        annotation, buffer_size=0
                    )  # no buffer for real annotations

        # Create output path
        filename = output / f"{task['name']}.png"

        if not overwrite and filename.exists():
            filename = output / f"{task['name']}_dup.png"

        image.save(filename, "PNG")

    # for debugging
    if num_rotated_rectangles > 0:
        print(f"Found {num_rotated_rectangles} rotated rectangles.")


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
@click.option(
    "--overwrite/--no-overwrite",
    type=bool,
    default=True,
    help="Whether to overwrite existing files in the output directory",
)
@click.option(
    "--buffer-size",
    type=int,
    default=10,
    help="The buffer size for the artificial boundaries in pixels. Set to 0 to disable.",
)
@click.option(
    "--considered-ids",
    type=str,
    default=None,
    help="Comma-separated list of task IDs to consider. If not set, all tasks are considered.",
)
@click.command
def labelstudio_to_png_cli(
    input, output, color, overwrite, buffer_size, considered_ids
):
    output.mkdir(exist_ok=True, parents=True)
    if considered_ids is not None:
        considered_ids = [int(i) for i in considered_ids.split(",")]
    labelstudio_to_png(
        input,
        output,
        color,
        overwrite,
        artboundary_buffer_size=buffer_size,
        considered_ids=considered_ids,
    )


if __name__ == "__main__":
    ecpo_data_to_labelstudio(
        pathlib.Path("../ecpo-data/data/annotator_data/JB-visualGT/"),
        pathlib.Path("labelstudio/labelstudio_data.json"),
        True,
    )
