# Ground Truth Strategy

This document tries to layout our full strategy regarding the handling of ground truth.
It is written down in order to identify any caveats before we start editing anything.
Otherwise, we might end up with annotation information being lost or very hard to transfer.

## What we have currently

The `ecpo-data` ground truth contains the following labels:
* `article`
* `image`
* `advertisement`
* `additional` (used for marginalia of the newspaper)

These labels have semantic meaning. Three of these are entirely or partially composed of
text blocks. This labelling reflects quite well what is the desired output of the overall
pipeline. However, it does not necessary comply with the requirements of all stages of
a multi stage processing pipeline.

Inspection of our groundtruth yielded that we have many segments that are labelled twice,
sometimes with a temporal distance of three years. Matthias does not know why and how this
happened - the working assumption is that it was a bug in the `ecpo-annotator` software.
In those cases, where the labels match, this does not matter. But we also have cases where
they do not match and it is interesting to study these to maybe get additional information.

* `article` + `image`: This typically indicates that a calligraphic heading. They annotated
  these as `image` with the reasoning being that OCR will not be able to read it because
  of the calligraphic nature. In a different labelling pass, it was labelled as article though.
* `image` + `advertisement`: Can be safely ignored, because `advertisement` is later deleted
  and `image` is correct on this, because it is sometimes used for images contained in ads.
* `additional` + other: These are only a few, that I have not seen visually.

Note: sometimes there are even triple labels, but they always contain a duplicate pair.

## Where we want to be

We need a ground truth that is `eynollah`-compatible. `eynollah` employs a multi-stage
pipeline that in its first stage uses the following labels:
* `text`
* `image`
* `heading`
* `separator`

## LabelStudio

With `ecpo-annotator` in a currently unusable state, we need an alternative labelling software.
We set up a [LabelStudio](https://labelstud.io/) instance at: https://label.ssc.uni-heidelberg.de/
Our strategy will involve converting our data to a format that imports into LabelStudio
and convert an available export format into something eynollah understands.

## Processing pipeline

### Step 1: Converting ecpo-data to LabelStudio

The conversion process is done by the `ecpodata2labelstudio` entrypoint of the
Python package. It takes the input directory and the desired output as parameters:

```bash
ecpodata2labelstudio --input ../ecpo-data/data/annotator_data/JB-visualGT/
```

While mapping all annotations from `ecpodata` to their counterpart in the
[LabelStudio JSON format](https://labelstud.io/blog/understanding-the-label-studio-json-format/#breaking-down-the-label-studio-json-format),
we are also correcting all IIIF URLs to new instance at `ecpo.cats.uni-heidelberg.de`.

We explicitly perform these modifications to the data:
* All double labellings with equal label are deduplicated by removing one annotation
* Duplicate labels with differing labels are mapped in the following way:
  * `image` + `article` -> `heading`
  * `additional` + something -> `text`
* Labels are renamed according to these rules:
  * `additional` -> `text`: All marginalia are text snippets at the end of the day.
  * `article` -> `text`
  * Remove `advertisement`, these need to be relabelled.

### Step 2: Labelling in LabelStudio

Our LabelStudio project will have the following labels:
* `text`
* `image`
* `heading`
* `separator`

Existing annotations are added as *predictions*, not *annotations* - meaning that if you
start annotating they are all there for you to correct, but only once you hit submit it
is changed to an *annotation*.

We should follow this check-list for labelling:

* [ ] Are there headings that are mislabelled as article?
* [ ] Label everything in advertisement sections correctly
* [ ] Mark all separators (rectangles faster?)
* [ ] Are all images actual images or are some headings?

### Step 3: Exporting from LabelStudio to Eynollah

As talked with Clemens on the call, we will use PNGs to feed into Eynollah. The alternative
would be PAGE-XML, but it seems a lot of work to translate to yet another rich metadata format,
just so that eynollah can convert it to such PNG afterwards. Also it allows us to deal with
overlapping annotations by assigning priority to labels (like layers in graphics processing).

We export (in the UI) as `JSON-MIN` and then convert to PNG:

```bash
labelstudio2png
```
