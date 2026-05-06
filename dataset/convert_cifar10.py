import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

# CIFAR-10 class names
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_images(data, labels, output_dir, start_idx=0):
    os.makedirs(output_dir, exist_ok=True)
    
    for i in tqdm(range(len(data))):
        img = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
        label = labels[i]
        class_name = classes[label]

        filename = f"{i+start_idx:05d}_{label}_{class_name}.png"
        filepath = os.path.join(output_dir, filename)

        Image.fromarray(img).save(filepath)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "cifar-10-batches-py")
    output_dir = base_dir

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    # Process training batches
    train_idx = 0
    for i in range(1, 6):
        batch = unpickle(os.path.join(input_dir, f"data_batch_{i}"))
        data = batch[b'data']
        labels = batch[b'labels']

        save_images(data, labels, train_dir, start_idx=train_idx)
        train_idx += len(data)

    # Process test batch
    test_batch = unpickle(os.path.join(input_dir, "test_batch"))
    save_images(test_batch[b'data'], test_batch[b'labels'], test_dir)

if __name__ == "__main__":
    main()
