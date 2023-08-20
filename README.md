# Electrical Box Inspection
This is a single python script that inspects the breaker, fuses, and temperature gauge of a given electrical box example. Provided are box examples along with reference images. 

The script will both display the outcome in the terminal and display images that coincide with the action the script is performing. If you want to see more in depth images of what's happening there are places in the script that can be uncommented.

## Setup:
First make sure you have python installed.
Then run the following commands in your terminal:
```
sudo apt-get install python3-pip
```
```
pip install opencv-python
```

## Running the Script:
```
python3 electrical_box_inspection.py 
```

## Box Examples:

The examples used were taken in a variety of lightings to try and make the program as accommodating as possible. Unit-less values have been used to make the program as adaptive as possible to current and additional boxes.

## Adding more Box Examples
More box examples can be added as the script is meant to work for different examples, however, more have not been tested. 

When adding examples you must add the box to the 'Box_Examples' with a given name that follows the pattern of the others, such as 'box_99'.

## Problems:

This is by no means a perfect script that works for every example. Lighting is a big factor, so drastic lighting can cause the program to not work properly.

The dial readings fro the temperature boxes will sometimes appear a few degrees off.