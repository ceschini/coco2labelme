# Coco to Labelme Converter

This repository holds an auxiliary script to convert coco annotations to labelme format.

This script allows the user to transform coco annotations to labelme format by simply specifying the ```coco.json``` file, the ```image folder``` and ```output folder```.

## Installation

Package requirements can be easilly installed via pip using the following command:

```python3
    pip install -r requirements.txt
```

Even though it can be installed in the global python environment, it is prefferable to install it in a virtual environment as follows:

1. Install ```virtualenv``` if you don't already have it.

    ```pip install virtualenv```

2. Create a new virtual environment via the ```virtualenv``` package.

```virtualenv .venv```

3.Install python packages requirements.

```pip install -r requirements.txt```

## Usage

Make sure you are on the correct folder with the appropriate files in it, and that the correct virtualenv is active, then, type on terminal:

```bash
python coco2labelme.py <coco_json> <data_path> <output_dir>
```

In order to run, please take notice of the three positional parameters.

* **coco_json**: coco.json file to be converted.
* **data_path**: Images data path, must match coco json filenames.
* **output_dir**: Output dir, must not previously exist.

## Contact

For questions and sugestions, you can [reach me on Zulip](https://chat.pixforcemaps.com/#narrow/pm-with/18-lucas.ceschini), or [e-mail me](mailto:lucas.ceschini@pixforce.ai).

***

Lucas Ceschini, Aug 29, 2022.
