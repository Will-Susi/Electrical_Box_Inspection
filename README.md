# Electrical Box Inspection
This is a single python script that inspects the breaker, fuses, and temperature gauge of a given electrical box. Provided are some examples of this box with various changes in lighting and states. 

The script will display the results it finds in the terminal. If you want to see a more in depth process of the program there are places in the script that can be uncommented that show the pictures of the steps.

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

- For users who are running WSL on Windows 10 or who are not running this program through a virtual machine, uncommenting the above mentioned lines will cause the program to not run.
- This is by no means a perfect script that works for every example, especially added ones; lighting is a big factor, so drastic lighting can cause the program to not work properly.
- The dial readings for the temperature boxes will sometimes appear a few degrees off.