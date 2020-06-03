# covid_19_prediction_with_deep_learning


Early in the epidemic, physicians were actually diagnosing cases of cornavirus using X-Ray and CT images. You can help them by developing more accurate methods for diagnosing COVID- 19 from chest X-Rays. Since COVID X-Rays are frequently confused with ordinary pneumonia, you will be asked to perform multi-class classification, distinguishing patients with COVID-19 from those who have viral and bacterial pneumonia or who are healthy.

The training data (available on Kaggle) includes 1127 chest xrays drawn from several different sources (of varying size and quality) and a set of multiclass labels indicating whether each patient was healthy or diagnosed with bacterial pneumonia, viral pneumonia, or COVID-19. The test data includes 484 images without labels for prediction.

In this project, I have explored implementing various Convolutional Neural Network (CNN) models on the basis of existing architecture like VGG16, ResNet50 using Keras library. Please see the complete training and testing dataset in the train and test directory which are also available in the kaggle competition page https://www.kaggle.com/c/4771-sp20-covid/overview
and the full code in the python file.