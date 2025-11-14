# Perspective Transform Demo

Interactive OpenCV utility that lets you drag four source points and view the resulting perspective warp in real time.

## Setup

Install dependencies via `pip` (Python 3.13+):

```bash
pip install -e .
```

## Usage

Provide an image path or omit it to open a file picker (when Tkinter is available):

```bash
python main.py path/to/image.jpg
```

Controls:

- Drag the four circles on the padded image to reshape the perspective source quad.
- Press `r` to reset the points to the original image corners.
- Press `Esc` or `q` to quit.
