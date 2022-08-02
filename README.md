# Electrical Box Inspection
This is a single python script that inspects the breaker, fuses, and temperature gauge of a given electrical box example. Provided are box examples along with reference images. 

The script will both display the outcome in the terminal and display images that coincide with the action the script is performing. If you want to see more in depth images of what's happening there are places in the script that can be uncommented.

## Setup
Python3 along with Pip3 needs to be installed.

File path locations inside the script need to be updated to their actual file paths for both the part examples and the box examples.

## Adding more box examples
More box examples can be added as the script is meant to work for different examples, however, more have not been tested. 

When adding examples you must add the box to the 'Box_Examples' folder along with adding its file path as another option at the beginning of the script:
```
elif box_num == '8': 
    box = 'image_path'
```
Depending on the example added, the script has only tested for two brightnesses. The references for both brightnesses are determined by hardstuck values so it is possible the box added won't work. To fix this make similar reference from the box image and add it to the 'Box_Parts' folder and add to the script like the other examples:

```
elif 'min_bright' < brightness and brightness < 'max_bright':
    breaker_reference = '../images/Box_Parts/breaker_1.jpg'
    fuse_reference = '../images/Box_Parts/fuses_1.jpg'
    temp_reference = '../images/Box_Parts/tempbox_1.jpg'
```

To get the min and max brightnesses, use print(brightness) with your new set box and set the 'min_bright' and 'max_bright' values below and above that, respectively.

## Running the script:
```
python3 electrical_box_inspection.py 
```
