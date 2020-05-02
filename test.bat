@echo off
rem Replace the variables with your github repo url, repo name, test video name, json named by your UIN
set GIT_REPO_URL=https://github.com/liangch0505/proj8.git
set REPO=proj8
set VIDEOPATH=singleTest
set UINJSON=430000801.json
set UINJPG=430000801.jpg
set JSON=timeLabel.json
set JPG=timeLabel.jpg
git clone %GIT_REPO_URL%
cd %REPO%
echo %VIDEOPATH%
python test.py --videopath_name %VIDEOPATH%
rem rename the generated timeLabel.json and figure with your UIN.
copy %JPG% %UINJPG%
copy %JSON% %UINJSON%