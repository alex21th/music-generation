
This folder includes several changes to the PyTorch [implementation](https://github.com/annahung31/MidiNet-by-pytorch) of [MidiNet](https://arxiv.org/abs/1703.10847).

The dataset is from [theorytab](https://www.hooktheory.com/theorytab) and can be downloaded [here](https://github.com/wayne391/Lead-Sheet-Dataset).

--------------------------------------------------------------------------------------------------
Prepare the data

get_data.py                     |  get melody and chord matrix from xml and json files

get_train_and_test_data.py      |  seperate the melody and chord data into training set and testing set

--------------------------------------------------------------------------------------------------
After you have the data, 
1. Make sure you have toolkits in the requirement.py
2. Run main.py ,  
  is_train = 1 for training, 
  is_draw = 1 for drawing loss, 
  is_sample = 1 for generating music after finishing training.
  
3. If you would like to turn the output into real midi for listening
  Run demo.py

--------------------------------------------------------------------------------------------------

We have included several models, which can be called from `main.py` simply changing the model to import.
