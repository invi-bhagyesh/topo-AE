import cv2
import os
import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Split or combine character images.")
parser.add_argument('--mode', choices=['split', 'combine'], default='split', help='Mode to run: split or combine')
parser.add_argument('--split_input', type=str, default='split', help='Input folder for split mode')
parser.add_argument('--split_output', type=str, default='combined', help='Output folder for split mode')
parser.add_argument('--combine_input', type=str, default='split', help='Input folder for combine mode')
parser.add_argument('--combine_output', type=str, default='combined', help='Output folder for combine mode')
parser.add_argument('--original_input', type=str, default='test', help='Folder containing original images for combine mode')
parser.add_argument('--padding', type=int, default=10, help='Padding to add around each character')
parser.add_argument('--char_size', type=int, default=64, help='Uniform size for character images (width and height)')
parser.add_argument('--debug', action='store_true', help='Save debug images showing detected character boundaries')
args = parser.parse_args()

if args.mode == "split":
    INPUT_FOLDER = args.split_input
    OUTPUT_FOLDER = args.split_output
    CHAR_SIZE = args.char_size
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    meta_path = os.path.join(OUTPUT_FOLDER, "metadata.txt")
    if os.path.exists(meta_path):
        os.remove(meta_path)

    # First pass: compute global max width and height of all valid contours in dataset
    global_max_w = 0
    global_max_h = 0

    for filepath in glob.glob(os.path.join(INPUT_FOLDER, "*.png")) + glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")):
        filename = os.path.basename(filepath)
        parts = filename.split("_")
        if len(parts) < 2:
            continue
        label = parts[1]

        img_color = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if img_color is None:
            continue
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        thresh = 255 - gray
        _, thresh_bin = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        valid_contours = []
        for ctr in contours:
            x, y, w, h = cv2.boundingRect(ctr)
            if w > 2 and h > 2:
                valid_contours.append(ctr)

        for i, ctr in enumerate(valid_contours):
            if i >= len(label):
                break
            x, y, w, h = cv2.boundingRect(ctr)
            if w > global_max_w:
                global_max_w = w
            if h > global_max_h:
                global_max_h = h

    print(f"Global max character size determined: width={global_max_w}, height={global_max_h}")

    pad = 5

    for filepath in glob.glob(os.path.join(INPUT_FOLDER, "*.png")) + glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")):
        filename = os.path.basename(filepath)
        print(f"Processing {filename}...")

        parts = filename.split("_")
        if len(parts) < 2:
            print(f"Skipping {filename}, filename format not matching")
            continue
        label = parts[1]

        img_color = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if img_color is None:
            print(f"Failed to load {filename}")
            continue
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        thresh = 255 - gray
        _, thresh_bin = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        valid_contours = []
        for ctr in contours:
            x, y, w, h = cv2.boundingRect(ctr)
            if w > 2 and h > 2:
                valid_contours.append(ctr)

        print(f" -> Found {len(valid_contours)} valid contours for '{label}' (expected {len(label)})")

        metadata = []
        h_orig, w_orig = gray.shape[:2]
        metadata.append(f"ORIGINAL_SIZE {h_orig} {w_orig}")

        for i, ctr in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(ctr)
            if i >= len(label):
                break

            char_img = img_color[y:y+h, x:x+w].copy()

            # Create white image of size (global_max_h, global_max_w)
            char_canvas = np.full((global_max_h, global_max_w, 3), 255, dtype=np.uint8)

            # Compute top-left corner to center the char_img in char_canvas
            y_offset = (global_max_h - h) // 2
            x_offset = (global_max_w - w) // 2

            char_canvas[y_offset:y_offset+h, x_offset:x_offset+w] = char_img

            # Add 5 pixel padding border around the image
            char_img_padded = cv2.copyMakeBorder(char_canvas, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255,255,255])

            char_label = label[i]
            out_path = os.path.join(OUTPUT_FOLDER, f"{filename[:-4]}_{i}_{char_label}.png")
            cv2.imwrite(out_path, char_img_padded)

            # Save global_max_w, global_max_h, and padding in metadata
            metadata.append(f"CHAR {i} {x} {y} {w} {h} {global_max_w} {global_max_h} {pad}")

        if args.debug:
            debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for i, ctr in enumerate(valid_contours):
                if i < len(label):
                    x, y, w, h = cv2.boundingRect(ctr)
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
                    cv2.putText(debug_img, label[i], (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"debug_{filename}"), debug_img)

        with open(meta_path, "a") as f:
            f.write(f"FILE {filename[:-4]}\n")
            for line in metadata[1:]:
                f.write(line + "\n")
            f.write(metadata[0] + "\n")
            f.write("END_FILE\n")

    print("✅ Done! Characters saved with uniform max size and padding. Metadata includes max_w, max_h, and padding.")
    final_size_w = global_max_w + pad * 2
    final_size_h = global_max_h + pad * 2
    print(f"🔎 Max character size (before padding): width={global_max_w}, height={global_max_h}")
    print(f"🖼️ Final saved character image size (with padding): width={final_size_w}, height={final_size_h}")

elif args.mode == "combine":
    INPUT_FOLDER = args.combine_input
    OUTPUT_FOLDER = args.combine_output
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    meta_file = os.path.join(INPUT_FOLDER, "metadata.txt")
    if not os.path.exists(meta_file):
        print("Error: metadata.txt not found in combine input folder")
        exit(1)

    print("Reading metadata from metadata.txt...")

    all_files_metadata = {}
    current_file = None

    with open(meta_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "FILE":
                current_file = parts[1]
                all_files_metadata[current_file] = {'original_size': None, 'chars': []}
            elif len(parts) >= 3 and parts[0] == "ORIGINAL_SIZE" and current_file:
                all_files_metadata[current_file]['original_size'] = (int(parts[1]), int(parts[2]))
            # Expect 9 tokens now (CHAR, idx, x, y, w, h, max_w, max_h, pad)
            elif len(parts) == 9 and parts[0] == "CHAR" and current_file:
                idx, x, y, w, h, max_w, max_h, pad = map(int, parts[1:])
                all_files_metadata[current_file]['chars'].append((idx, x, y, w, h, max_w, max_h, pad))
            elif parts == ["END_FILE"]:
                current_file = None

    for base_name, file_metadata in all_files_metadata.items():
        print(f"Processing {base_name}...")

        original_size = file_metadata['original_size']
        char_metadata = file_metadata['chars']

        if original_size is None:
            continue

        h_orig, w_orig = original_size
        combined_img = np.full((h_orig, w_orig, 3), 255, dtype=np.uint8)

        for idx, x, y, w, h, max_w, max_h, pad in char_metadata:
            pattern = os.path.join(INPUT_FOLDER, f"{base_name}_{idx}_*.png")
            char_files = glob.glob(pattern)
            if not char_files:
                continue

            char_img_padded = cv2.imread(char_files[0], cv2.IMREAD_COLOR)
            if char_img_padded is None:
                continue

            # Remove padding from padded char image (now directly the original char size)
            char_img_cropped = char_img_padded[pad:-pad, pad:-pad]

            # Crop center region corresponding to original bounding box (w, h)
            start_y = (char_img_cropped.shape[0] - h) // 2
            start_x = (char_img_cropped.shape[1] - w) // 2
            char_img = char_img_cropped[start_y:start_y+h, start_x:start_x+w]

            combined_img[y:y+h, x:x+w] = char_img

        out_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.png")
        cv2.imwrite(out_path, combined_img)
        print(f"✅ Saved combined image: {out_path}")

    print("✅ Done! All images combined back to original form.")

import cv2
import os
import glob
import numpy as np
from typing import List, Dict, Tuple, Any

def split_characters(dataloader, padding: int = 10, char_size: int = 64, debug: bool = False, debug_output_dir: str = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Split character images from dataloader into individual characters.
    
    Args:
        dataloader: Iterable that yields (image_array, filename, label) tuples
        padding: Padding to add around each character
        char_size: Uniform size for character images (width and height) 
        debug: Save debug images showing detected character boundaries
        debug_output_dir: Directory to save debug images (required if debug=True)
    
    Returns:
        Tuple of (character_dataloader_list, metadata_dict)
        - character_dataloader_list: List of dicts with keys 'image', 'filename', 'char_label', 'char_index'
        - metadata_dict: Dictionary containing all metadata needed for combining
    """
    
    if debug and debug_output_dir is None:
        raise ValueError("debug_output_dir must be provided when debug=True")
    
    if debug:
        os.makedirs(debug_output_dir, exist_ok=True)
    
    # Convert dataloader to list for two-pass processing
    data_list = list(dataloader)
    
    # First pass: compute global max width and height of all valid contours in dataset
    global_max_w = 0
    global_max_h = 0

    for img_array, filename, label in data_list:
        # Convert numpy array to opencv format if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_color = img_array.astype(np.uint8)
        else:
            img_color = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        thresh = 255 - gray
        _, thresh_bin = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        valid_contours = []
        for ctr in contours:
            x, y, w, h = cv2.boundingRect(ctr)
            if w > 2 and h > 2:
                valid_contours.append(ctr)

        for i, ctr in enumerate(valid_contours):
            if i >= len(label):
                break
            x, y, w, h = cv2.boundingRect(ctr)
            if w > global_max_w:
                global_max_w = w
            if h > global_max_h:
                global_max_h = h

    print(f"Global max character size determined: width={global_max_w}, height={global_max_h}")

    pad = 5
    character_dataloader = []
    all_files_metadata = {}

    for img_array, filename, label in data_list:
        print(f"Processing {filename}...")

        # Convert numpy array to opencv format if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_color = img_array.astype(np.uint8)
        else:
            img_color = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        thresh = 255 - gray
        _, thresh_bin = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        valid_contours = []
        for ctr in contours:
            x, y, w, h = cv2.boundingRect(ctr)
            if w > 2 and h > 2:
                valid_contours.append(ctr)

        print(f" -> Found {len(valid_contours)} valid contours for '{label}' (expected {len(label)})")

        metadata = []
        h_orig, w_orig = gray.shape[:2]
        metadata.append(f"ORIGINAL_SIZE {h_orig} {w_orig}")

        for i, ctr in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(ctr)
            if i >= len(label):
                break

            char_img = img_color[y:y+h, x:x+w].copy()

            # Create white image of size (global_max_h, global_max_w)
            char_canvas = np.full((global_max_h, global_max_w, 3), 255, dtype=np.uint8)

            # Compute top-left corner to center the char_img in char_canvas
            y_offset = (global_max_h - h) // 2
            x_offset = (global_max_w - w) // 2

            char_canvas[y_offset:y_offset+h, x_offset:x_offset+w] = char_img

            # Add 5 pixel padding border around the image
            char_img_padded = cv2.copyMakeBorder(char_canvas, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255,255,255])

            char_label = label[i]
            
            # Add to character dataloader
            character_dataloader.append({
                'image': char_img_padded,
                'filename': f"{filename}_{i}_{char_label}",
                'char_label': char_label,
                'char_index': i,
                'original_filename': filename
            })

            # Save global_max_w, global_max_h, and padding in metadata
            metadata.append(f"CHAR {i} {x} {y} {w} {h} {global_max_w} {global_max_h} {pad}")

        if debug:
            debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for i, ctr in enumerate(valid_contours):
                if i < len(label):
                    x, y, w, h = cv2.boundingRect(ctr)
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
                    cv2.putText(debug_img, label[i], (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            cv2.imwrite(os.path.join(debug_output_dir, f"debug_{filename}.png"), debug_img)

        # Store metadata for this file
        base_name = filename.replace('.png', '').replace('.jpg', '')
        all_files_metadata[base_name] = {
            'original_size': (h_orig, w_orig),
            'chars': []
        }
        
        for line in metadata[1:]:  # Skip the ORIGINAL_SIZE line
            parts = line.split()
            if len(parts) == 9 and parts[0] == "CHAR":
                idx, x, y, w, h, max_w, max_h, pad_val = map(int, parts[1:])
                all_files_metadata[base_name]['chars'].append((idx, x, y, w, h, max_w, max_h, pad_val))

    print("✅ Done! Characters processed with uniform max size and padding. Metadata includes max_w, max_h, and padding.")
    final_size_w = global_max_w + pad * 2
    final_size_h = global_max_h + pad * 2
    print(f"🔎 Max character size (before padding): width={global_max_w}, height={global_max_h}")
    print(f"🖼️ Final saved character image size (with padding): width={final_size_w}, height={final_size_h}")

    return character_dataloader, all_files_metadata


def combine_characters(character_dataloader: List[Dict[str, Any]], metadata: Dict[str, Any]) -> List[Tuple[np.ndarray, str]]:
    """
    Combine character images back into original form using metadata.
    
    Args:
        character_dataloader: List of character data from split_characters
        metadata: Metadata dictionary from split_characters
    
    Returns:
        List of (combined_image_array, filename) tuples
    """
    
    print("Processing character combination...")
    
    # Group character data by original filename
    chars_by_file = {}
    for char_data in character_dataloader:
        orig_filename = char_data['original_filename']
        base_name = orig_filename.replace('.png', '').replace('.jpg', '')
        if base_name not in chars_by_file:
            chars_by_file[base_name] = []
        chars_by_file[base_name].append(char_data)
    
    combined_images = []
    
    for base_name, file_metadata in metadata.items():
        print(f"Processing {base_name}...")

        original_size = file_metadata['original_size']
        char_metadata = file_metadata['chars']

        if original_size is None:
            continue

        h_orig, w_orig = original_size
        combined_img = np.full((h_orig, w_orig, 3), 255, dtype=np.uint8)

        # Get character data for this file
        file_chars = chars_by_file.get(base_name, [])
        
        for idx, x, y, w, h, max_w, max_h, pad in char_metadata:
            # Find the corresponding character image
            char_img_padded = None
            for char_data in file_chars:
                if char_data['char_index'] == idx:
                    char_img_padded = char_data['image']
                    break
            
            if char_img_padded is None:
                continue

            # Remove padding from padded char image (now directly the original char size)
            char_img_cropped = char_img_padded[pad:-pad, pad:-pad]

            # Crop center region corresponding to original bounding box (w, h)
            start_y = (char_img_cropped.shape[0] - h) // 2
            start_x = (char_img_cropped.shape[1] - w) // 2
            char_img = char_img_cropped[start_y:start_y+h, start_x:start_x+w]

            combined_img[y:y+h, x:x+w] = char_img

        combined_images.append((combined_img, f"{base_name}.png"))
        print(f"✅ Combined image: {base_name}.png")

    print("✅ Done! All images combined back to original form.")
    return combined_images
    
# Usage examples:
# Split mode with debug visualization:
# python3 script.py --mode split --split_input images --split_output characters --padding 10 --char_size 64 --debug

# Split mode without debug:
# python3 script.py --mode split --split_input ../wmadv --split_output characters --padding 10 --char_size 64

# Combine mode:
# python3 script.py --mode combine --combine_input characters --combine_output reconstructed --original_input ../wmadv