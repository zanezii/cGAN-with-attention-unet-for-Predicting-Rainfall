# cGAN-with-attention-unet-for-Predicting-Rainfall
cGAN with attention-unet for Predicting Rainfall

为了处理高分辨率图片，将`cGAN`应用到`U-net`上，并且把`PatchGAN`中的分辨器修替换进来；

*架构模型*：以`cGAN`为基础，`Generator`是`U-Net`，`Discriminator`是`PatchGAN`的分辨器；

### Process

1. ibraries for data manipulation, deep learning, and evaluation metrics. Some examples of these libraries are NumPy, Pandas, TensorFlow, Keras, Scikit-learn, and Matplotlib.

2. Load the radar reflectivity data from the 3092 .npy files in the 'example1' folder, and split them into training and testing datasets based on the 'radar_events_traing' and 'radar_events_test' text files, respectively. 

3. Convert the raw radar reflectivity data (dBz) to grayscale (0–255) using a suitable formula or function.

4. Scale the grayscale values of the radar reflectivity data to the range of 0–1 using the Min–Max scaler method, based on the min–max values from the training dataset.

5. Prepare the input and output data for the Transformer model. For the input data, use four latest radar reflectivity data (t-30, t-20, t-10 min, and t min) to predict radar reflectivity data 10 minutes ahead. For the output 11data, convert the predicted radar reflectivity data into precipitation using the Z–R relationship.

6. Develop a Transformer model using the TensorFlow and Keras libraries. You can try different hyperparameters and architectures to optimize the model's performance, based on the information you have provided.

7. Train the Transformer model on the training dataset, using the evaluation metrics (pearson correlation coefficient, root mean square error, nash-sutcliffe efficiency, critical success index, and fraction skill scores) to monitor the model's performance.

8. Evaluate the trained model on the testing dataset, using the same evaluation metrics as in step 7. Generate the required results, such as the comparison of the average values of the verification metrics for the 10-minute precipitation prediction of default models (U-net, convLSTM, cGAN, transformer, autoformer) in terms of the evaluation metrics, box plots of the verification metrics of model prediction at the time up to 90 minutes, FSS of model prediction at the time 10, 30, 60 minutes, and examples of precipitation at some forecast time.

9. Apply transfer learning to the previously trained model with effective cost-efficient computation to train the model.
   
### Evaluation

|              | cGan attention     | cGan                | ConvLSTM           | U-Net(性能最好….)  | pySTEPS             |
| ------------ | ------------------ | ------------------- | ------------------ | ------------------ | ------------------- |
| **R**        | 0.924807495232668  | 0.9142265834003557  | 0.8881974954304946 | 0.9281361138042048 | 0.7864782731004448  |
| **RMSE**     | 5.957104328781084  | 9.094442762857053   | 6.212152619657104  | 6.1628677229216    | 16.369856849649544  |
| **NSE**      | 0.734320992110765  | 0.38082767668015904 | 0.7110847997275789 | 0.7156509051050015 | -1.0062078287433631 |
| **CSI(0.1)** | 1.0                | 1.0                 | 1.0                | 1.0                | 1.0                 |
| **CSI(2.5)** | 0.8607687228305728 | 0.6449708636157884  | 0.8489547111539907 | 0.867986727333084  | 0.6011622587395774  |
| **CSI(5.0)** | 0.7510358231291129 | 0.687318502821934   | 0.7199995676653224 | 0.7693647139411589 | 0.5312585581933228  |

### 

# Version

### 20230531

##### 修改

修改unet模型的loss；

修改FSS数据保存方式；

##### 问题

读取数据时，generate_data()函数不知道如何处理数据；

pySTEPS预测时，有无效值；

pySTEPS预测时，leadtime为2可以看到部分正常图像，leadtime为15全有问题；



### 20230530

##### 修改

添加FSS数据的保存和画图；

修改FSS计算方式；

##### 问题

pySTEPS预测时，有无效值；



### 20230529

##### 修改

修改pySTEPS的运行bug；

##### 问题

FSS画图；

pySTEPS预测时，有无效值；

unet模型效果最优；



### 20230528

##### 修改

修改模型训练和结果评价的bug；

##### 问题

FSS画图；



### 20230527

##### 修改

添加pySTEPS对比模型；

修改模型训练时的label；

添加dbR与R的转换；

##### 问题

FSS画图；



### 20230526

##### 修改

添加U-net/ConvLSTM对比模型；

##### 问题

添加pySTEPS对比模型；



### 20230525

##### 修改

添加FSS评价函数；

修改画图函数；

修改预测时间；

##### 问题

添加U-Net/ConvLSTM/pySTEPS模型；



### 20230524

##### 修改

添加CSI评价函数；

##### 问题

添加FSS的评价函数；



### 20230522-2

##### 修改

修复performance函数取值的问题；

##### 问题

添加CSI/FSS的评价函数；



### 20230522

##### 修改

添加R/RMSE/NSE的评价函数；

##### 问题

添加CSI/FSS的评价函数；



### 20230520

##### 修改

Generator修改为attention-unet；

代码优化；

##### 问题

数据参数计算；



### 20230519

##### 修改

代码和论文分析完成；

##### 问题

添加注意力机制；



### 20230518

##### 修改

基础环境搭建完成，可以跑通；

##### 问题

添加注意力机制；
