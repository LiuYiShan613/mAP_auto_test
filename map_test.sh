#!/bin/bash

# set param
CONFIG_FILE="diva_test_config.yml"
# set video test dir path
VIDEO_DIR="/opt/nvidia/deepstream/deepstream-6.3/samples/streams/MOT"

# auto detect all mp4 files
VIDEOS=($(ls "${VIDEO_DIR}"/*.mp4))

# to find .mp4
if [ ${#VIDEOS[@]} -eq 0 ]; then
  echo "There's no .mp4 files in ${VIDEO_DIR} dir！"
  exit 1
fi

# go through every video
for VIDEO in "${VIDEOS[@]}"; do
  # update diva_test_config.yml (video list)
  sed -i "/^ *list: /s|list:.*|list: file://${VIDEO}|" "$CONFIG_FILE"

  # update resolution in diva_test_config.yml
  python3 update_config.py
  
  echo "testing video：$(basename "${VIDEO}")"
  
  # generate txt detect bbox
  ./diva-test-app "$CONFIG_FILE"
  
  echo "finish testing video：$(basename "${VIDEO}")"

done

# test generated txt (go to auto_test dir and exec cal_acc.py)=> test one model
# if [ -d "auto_test" ]; then
#   cd auto_test
#   python3 cal_acc.py
#   cd - > /dev/null 
# else
#   echo "Error: auto_test folder not found!"
#   exit 1
# fi

# test generated txt (go to auto_test dir and exec cal_acc.py) => if test all yolo models
if [ -d "auto_test" ]; then
  cd auto_test
  python3 cal_acc.py

  # clean all txt file in auto_test dir
  echo "Cleaning up auto_test folder..."
  rm -f *.txt
  echo "All .txt files in auto_test folder have been deleted."

  cd - > /dev/null 
else
  echo "Error: auto_test folder not found!"
  exit 1
fi




