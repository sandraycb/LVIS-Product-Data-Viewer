# <img width="167" height="143" alt="above_logo" src="https://github.com/user-attachments/assets/3491b9e5-083c-4bc6-b86c-8ecefab8660c" />  LVIS Product Data Viewer 
### View & explore data from LVIS collected during the [NASA Arctic Boreal Vulnerability Experiment (ABoVE)](https://above.nasa.gov/)  
#### Visualize lidar footprint waveforms and explore the lidar signal returns across a variety of terrain and vegetation land surfaces within flightlines aquired during 2017 & 2019 acquisitions across the North American boreal forest in Alaska and Canada.

| **Example: zoomed out LVIS flightline** | **Example: zoomed into LVIS flightline** | **Example: LVIS1B footprint waveform** |
| ------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------- |
| <img width="300" height="150" alt="CoarseZoom" src="https://github.com/user-attachments/assets/13063ca1-f644-47fe-939d-9f449da543e5" />  | <img width="300" height="150" alt="DetailedZoom" src="https://github.com/user-attachments/assets/0c57c7b4-966a-4629-83fa-8d23162915da" /> | <img width="300" height="150" alt="WaveformZoom" src="https://github.com/user-attachments/assets/ebfd358d-5cc6-4cfe-9864-db2f773f1f77" /> |


## Overview

The LVIS Product Data Viewer was created to allow LVIS users to interactively preview the LVIS L1B and L2 data without much coding experience or preprocessing necessary. 

## Requirements
- Python 3.7+
- Stable internet connection for map tiles

## Installation & Instructions
1. Clone or download this repository
2. Install python dependencies
3. Prepare your data:
    - Place LVIS L1B .h5 files in a dedicated folder
    - If using, place associated LVIS L2 files ('.TXT') in another folder
4. Start the server ('lvis-server.py')
5. When prompted, enter:
    - Path to L1B folder
    - Path to L2 folder (optional)
6. Open 'lvis-viewer.html' in your web browser
7. The viewer will automatically connect to 'http://localhost:5000'

## Demo Video
[![Watch the video](https://img.youtube.com/vi/I1WzewOrV8U/maxresdefault.jpg)](https://youtu.be/I1WzewOrV8U)

## Additional Resources
The lvis-tutorial.ipynb jupyter notebook contains a breakdown of accessing and visualizing data within the L1B and L2 files. 

## Acknowledgements
Code for this repository was developed with assistance from Claude (Anthropic) for initial code generation and debugging. Ideas are original to the authors, and all code has been reviewed, tested, and edited by them.  

##  
 Repository Contributors | Role | Affiliation | 
| ---------------- | ---------------- | ---------------- |
| Sandra Yaacoub | PI |  Dept. Geography & Planning, Queen's University
| Paul Montesano |  Contributor | NASA Goddard Space Flight Center ; ADNET Systems, Inc.|
