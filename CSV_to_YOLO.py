import pandas as pd

def convert_to_yolo_format(row):
    # YOLO format: <class_id> <center_x> <center_y> <width> <height>
    mode_to_class_id = {0: 0, 1: 1, 4: 2}
    
    class_id = mode_to_class_id.get(row['mode'], -1)
    
    if class_id == -1:
        return None  # Ignore rows with unsupported mode
    
    width = 4032  # Assuming the width of the images is 4032
    height = 3024  # Assuming the height of the images is 3024
    
    x_center = (row['x1'] + row['x2']) / (2 * width)
    y_center = (row['y1'] + row['y2']) / (2 * height)
    box_width = (row['x2'] - row['x1']) / width
    box_height = (row['y2'] - row['y1']) / height
    
    return f"{class_id} {x_center} {y_center} {box_width} {box_height}"

def csv_to_yolo(csv_file, output_folder):
    df = pd.read_csv(csv_file)
    
    for _, row in df.iterrows():
        yolo_format = convert_to_yolo_format(row)
        
        if yolo_format is not None:
            # Output YOLO format to a text file
            output_file_path = f"{output_folder}/{row['file'].split('.')[0]}.txt"
            with open(output_file_path, 'w') as output_file:
                output_file.write(yolo_format)

if __name__ == "__main__":
    csv_file_path = "./Annotations/training_file.csv"  # Replace with your CSV file path
    output_folder_path = "./Annotations/YOLO"  # Replace with your desired output folder path
    
    csv_to_yolo(csv_file_path, output_folder_path)
