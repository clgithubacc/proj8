# Instructions on testing on own video
<p>
Command: python --videopath_name PATH_TO_VIDEO_AND_OPENPOSE_JSONS<p>
This model takes OpenPose jsons as input. Under the project directory, please create a folder, and copy the video file into that folder. Then, create a folder named 'jsons' in that folder, and copy all OpenPose generated json files into that folder. Notice that video name should appear at the beginninng of json files in order to match json with video.<p>
If you are testing on a single video, then the output will be saved at the project directory (where test.py is). If there are multiple videos in the test folder, then the output will be saved in 'results' folder in the test folder.
 
## Code
For training, use proj8_train.py
For testing, use test.py

