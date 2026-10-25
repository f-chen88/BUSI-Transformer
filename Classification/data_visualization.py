import os
import matplotlib.pyplot as plt
from PIL import Image
from random import randint

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

dataset_base = os.path.join(project_root, 'Dataset_BUSI_with_GT')
normal_dir = os.path.join(dataset_base, 'normal')
benign_dir = os.path.join(dataset_base, 'benign')
malignant_dir = os.path.join(dataset_base, 'malignant')

def count_files_and_masks(path):
    total = 0
    file = 0
    mask = 0 

    for f in os.listdir(path) : 
        if 'mask' in f : 
            mask += 1
        else : 
            file += 1
        total += 1
    
    return total, file, mask

total_normal, file_normal, mask_normal = count_files_and_masks(normal_dir)
total_benign, file_benign, mask_benign = count_files_and_masks(benign_dir)
total_malignant, file_malignant, mask_malignant = count_files_and_masks(malignant_dir)

print("Normal : Total =", total_normal, ", Images =", file_normal, ", Masks =", mask_normal)
print("Benign : Total =", total_benign, ", Images =", file_benign, ", Masks =", mask_benign)
print("Malignant : Total =", total_malignant, ", Images =", file_malignant, ", Masks =", mask_malignant)


tab_normal = os.listdir(normal_dir)
tab_benign = os.listdir(benign_dir)
tab_malignant = os.listdir(malignant_dir)

fig = plt.figure()
for i in range (1, 19) : 
    ax = plt.subplot(3, 6, i)

    if i%2 != 0 : 
        num_normal = randint(1, file_normal)
        num_benign = randint(1, file_benign)
        num_malignant = randint(1, file_malignant)

    if i < 7 and i%2 != 0 : 
        files_normal = [file for file in tab_normal if file.startswith(f"normal ({num_normal})")]
        image_path = [file for file in files_normal if 'mask' not in file]
        path = os.path.join(normal_dir, image_path[0])
        image = Image.open(path)
        plt.imshow(image, cmap='gray')
        plt.title(f"Image ({num_normal})", fontsize=8)

    elif i < 7 and i%2 == 0 : 
        path_list = []
        files_normal = [file for file in tab_normal if file.startswith(f"normal ({num_normal})")]
        mask_path = [file for file in files_normal if 'mask' in file]
        for ele in mask_path : 
            path_list.append(os.path.join(normal_dir, ele))
        if len(path_list) == 2 : 
            path_1 = Image.open(path_list[0]).convert("L")
            path_2 = Image.open(path_list[1]).convert("L").resize(path_1.size)
            image = Image.blend(path_1, path_2, 0.5)
            plt.title(f"Mask ({num_normal})_double", fontsize=8)
        else : 
            image = Image.open(path_list[0])
            plt.title(f"Mask ({num_normal})", fontsize=8)
        plt.imshow(image, cmap='gray')
        

    if 6 < i < 13  and i%2 != 0 : 
        files_benign = [file for file in tab_benign if file.startswith(f"benign ({num_benign})")]
        image_path = [file for file in files_benign if 'mask' not in file]
        path = os.path.join(benign_dir, image_path[0])
        image = Image.open(path)
        plt.imshow(image, cmap='gray')
        plt.title(f"Image ({num_benign})", fontsize=8)

    elif 6 < i < 13 and i%2 == 0 : 
        path_list = []
        files_benign = [file for file in tab_benign if file.startswith(f"benign ({num_benign})")]
        mask_path = [file for file in files_benign if 'mask' in file]
        for ele in mask_path : 
            path_list.append(os.path.join(benign_dir, ele))
        if len(path_list) == 2 : 
            path_1 = Image.open(path_list[0]).convert("L")
            path_2 = Image.open(path_list[1]).convert("L").resize(path_1.size)
            image = Image.blend(path_1, path_2, 0.5)
            plt.title(f"Mask ({num_benign})_double", fontsize=8)
        else : 
            image = Image.open(path_list[0])
            plt.title(f"Mask ({num_benign})", fontsize=8)
        plt.imshow(image, cmap='gray')
        

    if i > 12  and i%2 != 0 : 
        files_malignant = [file for file in tab_malignant if file.startswith(f"malignant ({num_malignant})")]
        image_path = [file for file in files_malignant if 'mask' not in file]
        path = os.path.join(malignant_dir, image_path[0])
        image = Image.open(path)
        plt.imshow(image, cmap='gray')
        plt.title(f"Image ({num_malignant})", fontsize=8)


    elif i > 12 and i%2 == 0 : 
        path_list = []
        files_malignant = [file for file in tab_malignant if file.startswith(f"malignant ({num_malignant})")]
        mask_path = [file for file in files_malignant if 'mask' in file]
        for ele in mask_path : 
            path_list.append(os.path.join(malignant_dir, ele))
        if len(path_list) == 2 : 
            path_1 = Image.open(path_list[0]).convert("L")
            path_2 = Image.open(path_list[1]).convert("L").resize(path_1.size)
            image = Image.blend(path_1, path_2, 0.5)
            plt.title(f"Mask ({num_malignant})_double", fontsize=8)
        else : 
            image = Image.open(path_list[0])
            plt.title(f"Mask ({num_malignant})", fontsize=8)
        plt.imshow(image, cmap='gray')
    plt.axis('off')



# Row labels (placed outside the figure grid on the left)
fig.text(0.07, 0.78, "Normal", fontsize=16, va='center', ha='right', rotation=90)
fig.text(0.07, 0.51,  "Benign", fontsize=16, va='center', ha='right', rotation=90)
fig.text(0.07, 0.24, "Malignant", fontsize=16, va='center', ha='right', rotation=90)
    
fig.suptitle("Data visualization", fontsize=20)
plt.show()