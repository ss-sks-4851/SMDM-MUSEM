The project has 2 parts
1. Headline generation - ganText
This part is for headline generation.
Requirements
-Put the NELA dataset inside ganText/giga_word folder with name NELA_Train.
-python 3.7.11
-tensorflow 1.14.0

2. Paper architecture(attention model) - MuSem-main
This part is for attention model implementation.
Requirements
-Download the glove.6B.300d pretrained model and put in MuSem-main folder.
-python 3.9.7
-tensorflow 2.8.0

3. How to run the project

- Put the NELA dataset inside ganText/giga_word folder with name NELA_Train.
- Download the glove.6B.300d pretrained model and put in MuSem-main folder.
- Install the requirements of the respective part of project you want to run.

[*************Important*********]
- MuSem-main is the main part of the project. 
- To run the project go to the folder MuSem-main and run the command 
 1. python main.py

- To run the synthetic headline generation part. Go to folder ganText
 1. python make_pretrain.py
 2. python main.py -pretrain -num_steps 20000 
 3. python main.py -train -num_steps 7000  
 4. python main.py -test 

