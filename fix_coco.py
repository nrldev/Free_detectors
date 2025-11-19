import json

# The path to your annotation file
ann_file = '/home/jaa/Work/Prog/BSU/Detectors/projects/dfine/datasets/yolo_coco/annotations/instances_val.json'

try:
    # Reading JSON
    with open(ann_file, 'r') as f:
        data = json.load(f)

    # Updating categories
    for cat in data['categories']:
        if cat['id'] == 0:
            cat['id'] = 1
            print(f"Updated category: {cat}")

    # Updating annotations
    for ann in data['annotations']:
        if ann['category_id'] == 0:
            ann['category_id'] = 1
            print(f"Updated annotation: {ann['id']}")

    # Saving the updated file
    with open(ann_file, 'w') as f:
        json.dump(data, f, indent=2)
    print("File updated successfully!")

except FileNotFoundError:
    print(f"Error: File {ann_file} not found. Check the path.")
except json.JSONDecodeError:
    print("Error: Invalid JSON format in the annotation file.")
except Exception as e:
    print(f"Error: {str(e)}")
