First of All : Be sure that you are using GPU in colab.

🎯Source codes of the shoe sole detection project. I developed when I worked as an computer vision engineer at Eurobotik. Code Language is Pyhton. YOLO version is v8 (latest version during project development). I used Roboflow to pull the data sets. You can use your own dataset with api keys or use my datasets from roboflow(https://app.roboflow.com/yolov8-hlm3z/detectingshoesolesyolov8/3). I used version2 of my dataset in this repo but you can use whatever you want the main difference between second and third is that classes; v3 has only 1 classes which is soles despite v2 have 3 classes. The main idea about reducing class number is to be more specific for  soles. Pay attention to the paths used in the project, update according to your own project status.

🎯With prediction.py you can use the live webcam application on any IDLE. Don't forget to add the best.py file that we previously created in Google Colab.
![image](https://github.com/omertascioglu/YOLOv8-ShoeSoles-Detector/assets/33811400/c4b16591-170d-49db-84fe-fe84065afe82)

Angle.py = In this file, I took project one step further. In addition to the shoe sole, we can now estimate the angles.

<img width="980" alt="Ekran Resmi 2023-08-02 16 12 43" src="https://github.com/omertascioglu/YOLOv8-ShoeSoles-Detector/assets/33811400/94fa69bd-c766-4f8d-94d5-a106b0956fe0">

Alternative and modified live-time shoe sole angle and model finding

<img width="955" alt="Ekran Resmi 2023-08-10 10 23 46" src="https://github.com/omertascioglu/YOLOv8-ShoeSoles-Detector/assets/33811400/a8ffa9eb-4c43-44b6-b940-feb4cc89d893">
