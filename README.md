# covid_19_prediction_with_deep_learning


Early in the epidemic, physicians were actually diagnosing cases of cornavirus using X-Ray and CT images. You can help them by developing more accurate methods for diagnosing COVID- 19 from chest X-Rays. Since COVID X-Rays are frequently confused with ordinary pneumonia, you will be asked to perform multi-class classification, distinguishing patients with COVID-19 from those who have viral and bacterial pneumonia or who are healthy.

The training data (available on Kaggle) includes 1127 chest xrays drawn from several different sources (of varying size and quality) and a set of multiclass labels indicating whether each patient was healthy or diagnosed with bacterial pneumonia, viral pneumonia, or COVID-19. The test data includes 484 images without labels for prediction.

In this project, I have explored implementing various Convolutional Neural Network (CNN) models on the basis of existing architecture like VGG16, ResNet50 using Keras library. Please see the complete training and testing dataset in the train and test directory which are also available in the kaggle competition page https://www.kaggle.com/c/4771-sp20-covid/overview.


### Image Data Preprocessing

Data preprocessing on these CXR image data will be necessary to adapt them into the ResNet50 CNN model and achieve better performance during training. Hence, the following steps of data preprocessing have been implemented.
1. Resizing and Cropping
Since the given CXRs are drawn from different sources and are of different size and quality, the first step is to resize and crop these images into a same size and same shape, so that it can be input into the ResNet50 model. By applying keras preprocessing image library, I resized the data into 224 × 224 × 3 pixel values and stored them in arrays.
2. Normalization
Next, I normalize the data by scaling the pixel values from 0-255 down to 0-1 by dividing the values by 255. This helps ensure each input feature has similar data distribution and accelerate convergence when training the network. If this step is not taken, the convergence will take longer time since distribution of different feature values will likely be different, which complicate the learning process of the network.
3. Onehot Encoding on Labels
To build the multi-class classifier, I also applied the one-hot encoding on the label column to generate four separate label columns each representing one of the four classes ’bacterial’, ’viral’, ’normal’ and ’covid’ . The last layer will use the activation function ’softmax’ to output the class with the greatest probability. Without this step, the model will not be able to handle single-column four-class output layer.
 4. Data Augmentation
Lastly, I tried implementing the data augmentation with the ImageDataGenerator tool in Keras, by randomly rotating the image within 20 degrees and shifting for 10% in width and height. This is in an attempt to expose the network to a wider variations so that the model is less prone to undesired characteristics, for example, the angle of the CXR taken, and thus is less likely to be overfitting. This data augmentation has helped stablized the classifier, which has lowered training accuracy increase rate during epochs but improved the accuracy on testing test for 3%.
In figure 1, the resized representative chest X-ray images for bacterial pneumonia, covid-19, normal, viral pneumonia are displayed respectively.

<img src="https://github.com/leyimai/covid_19_prediction_with_deep_learning/blob/master/report_figures/1samples.png" width="600"  />


### Model Training & Error Analysis

Based on the ResNet50 architecture, the model was modified to include a total of 53 convo- lutional layers in total, the first using a filter of size 7 × 7 and the latter using a filter of size 3 × 3. Batch normalization layers and spatial dropout layers were implemented intermittently to prevent overfitting.

The next step was to tune the hyperparameters of the model. To see which sets of hy- perparamters lead to the most stable best result, I tried different combinations of crucial hyperparameters, including batch size, number of epochs, dropout rate and optimizer learn- ing rate. To evaluate the performances of each model, I used 10% of data as validation set and plotted the validation accuracy curve and loss curve. As a result, the model accuracy was improved from 70% to 77% by setting the batch size as 32, number of epochs as 100, dropout rate as 0.2 and learning rate as 0.0005.

As is shown in Figure 2, the model has reached accuracy of approximately 73% on both the training and testing set, with the the loss on both sets converged nicely at the later epochs. In figure 2, the learning curves of validation accuracy and loss are shown.

<img src="https://github.com/leyimai/covid_19_prediction_with_deep_learning/blob/master/report_figures/2learning_curve.png" width="500"  />

### Model Evaluation Metrics on Test Set
<img src="https://github.com/leyimai/covid_19_prediction_with_deep_learning/blob/master/report_figures/3metrics.png" width="500"  />

### Presion-recall curve and ROC curve
<img src="https://github.com/leyimai/covid_19_prediction_with_deep_learning/blob/master/report_figures/4performance_curve.png" width="600"  />

### Visualization of the Model
<img src="https://github.com/leyimai/covid_19_prediction_with_deep_learning/blob/master/report_figures/interpretation2.png" width="600"  />

