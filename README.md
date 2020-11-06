# _**StayFocus**_

## Online Learning Engagement Using Facial Expression

### A. Description of the project

#### i. Project Description
![Diagram](./StayFocus.png)


#### ii. Input and output information
- **Input:** Webcam as video input
- **Output:** Classification (emotion) with Localization (bounding box)

### B. Motivation of the idea
- Currently, there are a lot of online learning done by schools and universities due to the pandemic. Therefore, we think by doing this project, it will help the teachers monitor the students of their engagement during the lessons.
- This project also our CDLE capstone project.

### C. Data Set Sources
- **Dataset**: DAiSEE dataset. The dataset can be download [here](https://iith.ac.in/~daisee-dataset/).

```
A Gupta, A DCunha, K Awasthi, V Balasubramanian
DAiSEE: Towards User Engagement Recognition in the Wild
arXiv preprint: arXiv:1609.01885
```
- 4 classes :
    - Boredom
    - Confusion
    - Engagement
    - Frustration

### D. Network Description
*Face Detection and Localisation*

- Caffe model using OpenCV’s DNN module

*Classification*
- Pretrained model: VGG 16
- Fine tuning: fc2, predictions layer

### E. Model Training

*Dataset Preprossessing*
1. Randomly select 12 videos per class [Very High (3) level]
2. Extract 30 frames per video
3. Total 360 images per class in the dataset

*Training Details*
- Split dataset into 70:30 ratio (70% training set)
- Train for 20 epochs on CPU (i5 9th gen)
- Duration: ~3 hours
- Accuracy: 73.51%

### F. Testing
- Remaining 30% from the dataset split
- Accuracy: 78.94%

### G. Future Development
- Gesture Detection
    - To include body gesture into the model
- Support Message
    - To give a message to student to regain focus in the lesson
- Engagement Gauge
    - To give feedbacks to teachers about the students' focus level

### H. List of Group members
- Atiff Zakwan, atiffzakwan@gmail.com
- Soo Wan Yong, wanyong_soo@hotmail.com
