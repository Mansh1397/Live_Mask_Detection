# Live_Mask_Detection

A pytohn based project which works on detecting whether or not a mask has been worn in a LIVE video. 
  * For live webcam view: opencv
  * For Modelling: CNN
  
To run this code, simply run master.py with first argument being "train" if you want to train your model and "detect" if you have a trained model and want to test on live webcam.
Even if you opt "detect" without any saved model, the code will train the model first and then procees for detection.
The second system argument is whether you want to opt for early stopping or not. It is a boolean, i.e. True/False and it is an optional argument.
The runtime command should look like this:

  python master.py "train" "True"
  
           OR

  python master.py "detect" "True"
  
Note: The "True" in the above syntax could be ignored.
