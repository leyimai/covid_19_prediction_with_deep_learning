# covid_19_prediction_with_deep_learning


Early in the epidemic, physicians were actually diagnosing cases of cornavirus using X-Ray and CT images. You can help them by developing more accurate methods for diagnosing COVID- 19 from chest X-Rays. Since COVID X-Rays are frequently confused with ordinary pneumonia, you will be asked to perform multi-class classification, distinguishing patients with COVID-19 from those who have viral and bacterial pneumonia or who are healthy.

The training data (available on Kaggle) includes 1127 chest xrays drawn from several different sources (of varying size and quality) and a set of multiclass labels indicating whether each patient was healthy or diagnosed with bacterial pneumonia, viral pneumonia, or COVID-19. The test data includes 484 images without labels for prediction.

In this project, I have explored implementing various Convolutional Neural Network (CNN) models on the basis of existing architecture like VGG16, ResNet50 using Keras library. Please see the complete training and testing dataset in the train and test directory which are also available in the kaggle competition page https://www.kaggle.com/c/4771-sp20-covid/overview.


### Representative chest X-ray images
<img src="https://github.com/leyimai/covid_19_prediction_with_deep_learning/blob/master/report_figures/1samples.png" width="600"  />


### Model Training & Error Analysis

Based on the ResNet50 architecture, the model was modified to include a total of 53 convo- lutional layers in total, the first using a filter of size 7 × 7 and the latter using a filter of size 3 × 3. Batch normalization layers and spatial dropout layers were implemented intermittently to prevent overfitting.

The next step was to tune the hyperparameters of the model. To see which sets of hy- perparamters lead to the most stable best result, I tried different combinations of crucial hyperparameters, including batch size, number of epochs, dropout rate and optimizer learn- ing rate. To evaluate the performances of each model, I used 10% of data as validation set and plotted the validation accuracy curve and loss curve. As a result, the model accuracy was improved from 70% to 77% by setting the batch size as 32, number of epochs as 100, dropout rate as 0.2 and learning rate as 0.0005.

As is shown in Figure 2, the model has reached accuracy of approximately 73% on both the training and testing set, with the the loss on both sets converged nicely at the later epochs. In figure 2, the learning curves of validation accuracy and loss are shown.

Figure 2
<img src="https://github.com/leyimai/covid_19_prediction_with_deep_learning/blob/master/report_figures/2learning_curve.png" width="500"  />

### Model Evaluation Metrics on Test Set
<img src="https://github.com/leyimai/covid_19_prediction_with_deep_learning/blob/master/report_figures/3metrics.png" width="500"  />

### Presion-recall curve and ROC curve
<img src="https://github.com/leyimai/covid_19_prediction_with_deep_learning/blob/master/report_figures/4performance_curve.png" width="600"  />

### Visualization of the Model
<img src="https://github.com/leyimai/covid_19_prediction_with_deep_learning/blob/master/report_figures/interpretation2.png" width="600"  />

