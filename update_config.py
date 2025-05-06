import cv2
import yaml

# Function to extract video resolution
def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Unable to open video file: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

# Function to update YAML file
def update_yaml_file(yml_path, width, height):
    with open(yml_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update streammux dimensions
    config["streammux"]["width"] = width
    config["streammux"]["height"] = height

    # Update tiler dimensions
    config["tiler"]["width"] = width
    config["tiler"]["height"] = height

    # Write updated values back to the file
    with open(yml_path, 'w') as file:
        yaml.safe_dump(config, file)

def main():
    yml_file_path = "diva_test_config.yml"

    try:
        # Load YAML file
        with open(yml_file_path, 'r') as file:
            config = yaml.safe_load(file)

        # Extract video path from YAML
        video_path = config["source-list"]["list"]
        video_path = video_path.replace("file://", "")  # Remove "file://" prefix

        # Get video resolution
        width, height = get_video_resolution(video_path)
        print(f"Video Resolution: {width}x{height}")

        # Update YAML file
        update_yaml_file(yml_file_path, width, height)
        print("YAML file updated successfully.")

    except Exception as e:
        print(e)
        return -1

if __name__ == "__main__":
    main()

