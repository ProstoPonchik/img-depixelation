## Image Depixelation Project Documentation

### Overview

This project is designed to depixelate images using machine learning. The primary objective is to predict the original values of pixelated regions within images.

### Setup

1. **Clone the Repository**
   ```bash
   git clone [<repository_url>](https://github.com/ProstoPonchik/img-depixelation.git)
   ```

2. **Install Dependencies**
   - Ensure you have the necessary Python libraries and dependencies installed. This project is optimized for NVIDIA GPUs with CUDA cores. It does not work on CPU by default, but you can modify it if required.
   - Install the required libraries from `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

3. **Download and Extract Data**
   - Download `images_for_training.zip` from the project's GitHub repository.
   - Extract the images:
     ```bash
     unzip images_for_training.zip
     ```

4. **Set Up Training Data Path**
   - Before training, you need to specify the path to your images in the `train_model.py` file:
     ```python
     dataset = RandomImagePixelationDataset(
         r"your_path_to_images",
         width_range=(4, 32),
         height_range=(4, 32),
         size_range=(4, 16)
     )
     ```
   Replace `your_path_to_images` with the path to your extracted images.

### Training the Model

1. **Train the Neural Network**
   Use the `train_model.py` script to train your model. As mentioned, it's optimal to run this on NVIDIA GPUs with CUDA cores for performance reasons.
   ```bash
   python3 train_model.py
   ```

### Testing the Model

1. **Generate Predictions**
   Use the `test_model.py` script to see the results of your trained model. Example usage is provided within the file.

2. **Evaluating Model Accuracy with Test Set**
   - First, you'll need to serialize your predictions using `submission_serialization.py`:
     ```python
     from a7_ex1 import produce_predictions
     predictions = produce_predictions("v4/model.pth", "v4/best_model_weights_validation.pth", "test_set.pkl")
     serialize(predictions, 'pudg.bin')
     ```
   - Then, use the provided `scoring.py` script to compute the RMSE:
     ```bash
     python3 scoring.py --submission ****.bin --target target.data
     ```

### Important Notes

- Always ensure you're using the correct image sets. `images_for_training.zip` should be used for training, while `test.zip` is for model accuracy testing.
- The project is optimized for NVIDIA GPUs with CUDA cores due to performance benefits. If you attempt to run it on a CPU, modifications may be needed.
- Use the provided scripts as outlined to ensure correct model training, testing, and scoring.
- The project automatically converts images to the required format.

### Troubleshooting

If you encounter any issues or have questions about specific parts of the project, refer to the detailed instructions provided in the original assignment or reach out to the project maintainers on GitHub.

---

This documentation provides a comprehensive guide on how to use the Image Depixelation project hosted on GitHub. Always ensure you follow the steps sequentially and refer back to the documentation if you encounter any challenges.
