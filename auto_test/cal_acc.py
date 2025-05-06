from DIVA_measure import DIVA_info, DIVA_gt
import cv2
from collections import deque
import os
import re

def parse_file_continuous(file_path):
    all_results = []
    current_res_list = []

    with open(file_path, "r") as file:
        for line in file:
            # split test when meet "Stream num"
            if line.startswith("Stream num:"):
                if current_res_list:
                    all_results.append(current_res_list)
                    current_res_list = []
                continue
            
            try:
                bbox_data = list(map(int, line.strip().split(",")))
                if len(bbox_data) == 4:  
                    current_res_list.append(bbox_data)
            except ValueError:
                continue  
        
        if current_res_list:
            all_results.append(current_res_list)

    return all_results


# extract yolo name
def extract_model_name(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        match = re.search(r'onnx-file: .*?/(\w+)\.pt\.onnx', content)
        if match:
            return match.group(1)
        else:
            return "Model name not found"
    except FileNotFoundError:
        return "File not found"


def main():
    
    # AI model name
    yml_file_path = '../diva_test_pgie_config.yml'
    model_name = extract_model_name(yml_file_path)
    print("model_name : ", model_name)
    
    video_dir = "/opt/nvidia/deepstream/deepstream-6.3/samples/streams/MOT"
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
    if not video_files:
        print(f"no .mp4 videos in {video_dir} dir")
        return

    for path in video_files:
        print(f"now processing: {path}")
    
        #raed file
        file_path = os.path.splitext(os.path.basename(path))[0] + ".txt"
        # file_path = "output_.txt"  
        all_results = parse_file_continuous(file_path)

        # path = "file:///opt/nvidia/deepstream/deepstream/samples/streams/MOT/MOT16-07.mp4"
        # DIVA video capture lib init
        cap = cv2.VideoCapture(path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        DV = DIVA_info(width, height)
        GT = DIVA_gt(path)

        ACC = 0
        for i, res_list in enumerate(all_results):
            DV.Frame_count += 1

            ret, frame = cap.read()
            if not ret:
                break

            # save draw result
            # result_dir = os.path.splitext(os.path.basename(path))[0]
            # if not os.path.exists(result_dir):
            #     os.makedirs(result_dir)
            # output_path = os.path.join(result_dir, f'frame_{DV.Frame_count:04d}.png')

            # draw ground truth box
            # GT.draw_gt(frame, DV.Frame_count-1, False)

            # draw detect
            # for bb in res_list:
            #     x_min, y_min, x_max, y_max = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
            #     cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255,0,0), 5)
            
            # save to dir
            # if file_path == "MOT20-05.txt" or file_path == "MOT20-03.txt" or file_path == "MOT17-05-DPM.txt":
            #     cv2.imwrite(output_path, frame)
            # print('res_list :', res_list)
            # print('-----------', len(res_list))
            
            DV.get_info(res_list, 0, 0)
            acc = GT.calc_acc(res_list, DV.Frame_count-1, False)
            ACC += acc
            # break
            # to calculate acc
            # if DV.Enable_calc:
            # acc = GT.calc_acc(res_list, DV.Frame_count-1, False)
            # DV.Frame_count += 1
            
        ACC_ans = ACC/DV.Frame_count
        print(f"ACC = {ACC_ans}")
        DV.calc_info(GT.gt_acc, GT.gt_number-1)
        DV.save_to_excel(file_path, ACC_ans, model_name)

if  __name__ == "__main__":
    main()
