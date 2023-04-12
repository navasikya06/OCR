# OCR

An OCR project using different deep learning models (Densenet-CTC and Differential Binerization) to detect and recognize a series and turn into text

* The project involves generating mock images similar to the sample ones. Series of characters and numbers are generated based on the rules given and imposed onto an image with different texts and backgrounds. These mock images are used together with the sample ones for the training of deep learning models.

* The main deep learning model used is Densenet-CTC. We also explored the differential binerazation paper using ResNet for object detection. 

For prediction: the code takes the image, augments it for processing before passing it through the stored trained models for prediction and final output of the text.


