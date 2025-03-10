The whole project is split into getting extracting the data from a mp4 video and actually predicting boulder grades based on the data.

Demo video:
https://github.com/user-attachments/assets/f74f4036-2121-4f26-aed3-695bb7a70d6e


New updates bring new stuff. The newest addition is the "Main_live_demo.py" which opens the camera, searches for kilterboard app images to align and predicts the grade. 
It does not simply predict on the image but creates a virtual replica which it creates by searching for rings in the original image. It feeds the replica into the trained CNN and outputs the prediction. It works ok right now but the prediction could be better.

Two models are saved. Best model so far is the climbing_grade_cnn_aug.pth.


I will update the whole repo over time with new things and ideas. 
With the adequate installs, the Main should run out of the box. If not, I will dockerize the whole process in due time.


To implement: 
              
              trainsition from CNN to GNN -> big step for every part of the process, so bound to take a while

              Upload the mp4 for dataextraction via Github LSF (its a 20 minute video)

              Use Docker
              
              
                 
  With this I invite all of you to try with me and make this a really good method of predicting Kilterboard grades.
