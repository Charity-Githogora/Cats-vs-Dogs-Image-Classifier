CATS VS DOGS IMAGE CLASSIFIER (🐱 VS 🐶)

This is a convolutional neural network (CNN) built with TensorFlow/Keras to classify images of cats and dogs.

Project Overview  
This project was completed as part of freeCodeCamp's Machine Learning with Python Certification. The goal was to build a CNN model that correctly classifies images of cats and dogs with at least 63% accuracy (achieved 66%).  

Key Features
✔ Data Augmentation: Used `ImageDataGenerator` to reduce overfitting.  
✔ CNN Architecture: Built with `Conv2D`, `MaxPooling2D`, and `Dense` layers.  
✔ Training/Validation: Evaluated performance using validation splits.  
✔ Test Predictions: Visualized model predictions on unseen data.  

---

Technologies Used
- Python   
- TensorFlow / Keras  
- OpenCV (for image preprocessing)  
- Matplotlib (visualization)  

---

Dataset  
The dataset was provided by freeCodeCamp and structured as:  
```
cats_and_dogs/
├── train/          # 2000 images (1000 cats, 1000 dogs)
│   ├── cats/
│   └── dogs/
├── validation/     # 1000 images (500 cats, 500 dogs)
│   ├── cats/
│   └── dogs/
└── test/           # 50 unlabeled images
```

---

How to Run the Code  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/cats-vs-dogs-classifier.git
   cd cats-vs-dogs-classifier
   ```

2. Install dependencies:  
   ```bash
   pip install tensorflow matplotlib numpy
   ```

3. Open the Jupyter Notebook:  
   ```bash
   jupyter notebook cats_vs_dogs_classifier.ipynb
   ```

4. Run all cells to train the model and see predictions.  

---

Results  
- Training Accuracy: 66%  
- Validation Accuracy: 66.4%  


---

Improvement Ideas (Optional)
🔹 Try 'transfer learning' with models like MobileNetV2.  
🔹 Experiment with 'hyperparameter tuning' (e.g., learning rate, batch size).  
🔹 Add 'Dropout layers' to reduce overfitting further.  

---

Credits 
- Project instructions by [freeCodeCamp](https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/cat-and-dog-image-classifier).  
- Dataset sourced from [Kaggle Cats vs Dogs](https://www.kaggle.com/c/dogs-vs-cats).  

---

License
This project is open-source under the [MIT License](LICENSE).  

---
