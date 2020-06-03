# covid_19_prediction_with_deep_learning


Early in the epidemic, physicians were actually diagnosing cases of cornavirus using X-Ray and CT images. You can help them by developing more accurate methods for diagnosing COVID- 19 from chest X-Rays. Since COVID X-Rays are frequently confused with ordinary pneumonia, you will be asked to perform multi-class classification, distinguishing patients with COVID-19 from those who have viral and bacterial pneumonia or who are healthy.

The training data (available on Kaggle) includes 1127 chest xrays drawn from several different sources (of varying size and quality) and a set of multiclass labels indicating whether each patient was healthy or diagnosed with bacterial pneumonia, viral pneumonia, or COVID-19. The test data includes 484 images without labels for prediction.

In this project, I have explored implementing various Convolutional Neural Network (CNN) models on the basis of existing architecture like VGG16, ResNet50 using Keras library. Please see the complete training and testing dataset in the train and test directory which are also available in the kaggle competition page https://www.kaggle.com/c/4771-sp20-covid/overview.


### 1. Image Data Preprocessing

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

<style>
img {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
 
<img src="https://github.com/leyimai/covid_19_prediction_with_deep_learning/blob/master/report_figures/1samples.png" width="600"  />


### 2. Model Training

Based on the ResNet50 architecture, the model was modified to include a total of 53 convo- lutional layers in total, the first using a filter of size 7 × 7 and the latter using a filter of size 3 × 3. Batch normalization layers and spatial dropout layers were implemented intermittently to prevent overfitting.

The next step was to tune the hyperparameters of the model. To see which sets of hy- perparamters lead to the most stable best result, I tried different combinations of crucial hyperparameters, including batch size, number of epochs, dropout rate and optimizer learn- ing rate. To evaluate the performances of each model, I used 10% of data as validation set and plotted the validation accuracy curve and loss curve. As a result, the model accuracy was improved from 70% to 77% by setting the batch size as 32, number of epochs as 100, dropout rate as 0.2 and learning rate as 0.0005.

As is shown in Figure 2, the model has reached accuracy of approximately 73% on both the training and testing set, with the the loss on both sets converged nicely at the later epochs. In figure 2, the learning curves of validation accuracy and loss are shown.

<img src="https://github.com/leyimai/covid_19_prediction_with_deep_learning/blob/master/report_figures/2learning_curve.png" width="500"  alt="centered image"/>

### 3. Error Analysis

Conducting a detailed error Analysis would also be helpful and meaningful in evaluating the model performance and its implication for clinical setting. The model evaluation metrics, confusion matrix and performance curves of this particular model are shown in Figure 3, Figure 4 and Figure 5, respectively.

As is shown in these results, the model achieves a 76% accuracy on the test set and an almost 100% recall rate (100% on test set and 98.7% on training set). From the confusion matrix, we can see that viral and viral pneumonia are sometimes confused with each other, and normal cases are sometimes classified as bacterial pneumonia or covid.

Considering the covid background, this is a relatively satisfactory result, since failing to identify a covid-19 case is dangerous for the fast spread of covid-19 virus. Although we do not want to falsely identify negative covid-19 case to be positive which would worsen the medical resource shortage, identifying true positive is so important and crucial that we can accept a certain level of false positive cases. The point is that if the recall rate of this classifier is trained to be high enough, it could assist physicians in the diagnosis of COVID-19 with the help of Chest X-rays, especially when the testing kit supply is falling short of the demand.

Therefore, in this COVID-19 setting, we would prefer a model with higher recall or true positive rate, even with a slight drop in precision and slight increase in false positive rate in the tradeoff between recall and precision, as well as between true positive rate and false positive rate.

<img src="https://github.com/leyimai/covid_19_prediction_with_deep_learning/blob/master/report_figures/3metrics.png" width="500"  />

<img src="https://github.com/leyimai/covid_19_prediction_with_deep_learning/blob/master/report_figures/3cm.png" width="500"  />

<img src="https://github.com/leyimai/covid_19_prediction_with_deep_learning/blob/master/report_figures/4performance_curve.png" width="600"  />


### 4. Model Interpretability

While aiming for higher accuracy and recall is one of the main goals, the model’s inter- pretability is also important, which would provide insight on what features are guiding the model’s prediction in classifying an CXR image. Of the models that I explored, Random forest and Support Vector Machine are generally models with good interpretability. The idea of using multiple decision trees to vote or finding the optimal boundaries of these mod- els make them easier to interpret, with feature importance being more comparable and the decision process being more transparent and understandable.

In comparison, neural network is relatively hard to interpret. The hidden inter-connected layers are like a black box. Nevertheless, there are some tools that we could explore to visu- alize the convolutional neural network so that we can better interpret what are the features extracted and used to distinguish between different classes.

In Figure 6a, the output of four main blcoks are visualized, which shows different features that are extracted at different layers. We can see that the starting layers look at the low-level features including the edges and shade of the lung , whereas the later layers look at higher- level features like some specific positions inside lungs. The visualization of the layerwise output could help us understand the hidden processing of the model to see which pixel areas are crucial in predicting the classes.

In Figure 6b, I used Local Interpretable Model-Agnostic Explanations (LIME) to explain the predictions of the Convolutional Neural Network classifier that I trained. Lime works by perturbing the features in an example and fitting a linear model to determine which features were most contributory to the model’s prediction for that example. As is seen in Figure 6b, superpixels in green indicate regions that were most contributory toward the predicted class, while superpixels coloured red indicate regions that were most contributory against the predicted class. This methods provide a very clear insight for physicians which areas inside the lung they should pay special attention to in the diagnosis of COVID-19.


<img src="https://github.com/leyimai/covid_19_prediction_with_deep_learning/blob/master/report_figures/interpretation2.png" width="600"  />

