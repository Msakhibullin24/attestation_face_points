# Facial Keypoint Detection Project

This project is designed to detect facial keypoints from an image.

## How to Test

1.  **Set up the environment:**
    *   Ensure you have Python installed.
    *   It is recommended to use a virtual environment.
    ```bash
    python -m venv myvenv
    source myvenv/bin/activate
    ```
    *   Install the required dependencies:
    ```bash
    pip install -r requirements.txt 
    ```
    *(Note: A `requirements.txt` file is not present. You may need to create one based on the project's imports.)*

2.  **Train the model (if necessary):**
    *   The training data is expected in `training.csv`.
    *   Run the training script:
    ```bash
    python train.py
    ```

3.  **Run the main application:**
    *   The main script will process an image and display the results.
    ```bash
    python main.py
    ```

## Example Result

Here is an example of the model's output on a test image:

![Project Result](image.png)
