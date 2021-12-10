import shutil
from os import path
import os

number = 4

while number <= 9:
    png_count = 0
    for i in range(0, 60000):
        source_path = f"D:/python/Dataset/mnist_png/training/{number}/{i}.png"
        if path.exists(source_path):
            png_count += 1

    split_number = png_count // 5
    print(f'Цифра {number}')
    print(f'Количество изображений в папке {split_number}')
    path_name = 1

    while path_name <= 5:
        print(f'Папка mnist-{path_name}')
        destination_path = f"D:/python/Dataset/mnist-{path_name}/{number}"

        i = 0
        for j in range(0, 60000):
            if i == split_number:
                break
            source_path = f"D:/python/Dataset/mnist_png/training/{number}/{j}.png"
            # print(source_path)

            if path.exists(source_path):
                shutil.move(source_path, destination_path)
                # print("% s перемещен в указанное место,% s" % (source_path, new_location))
                i += 1
            else:
                continue
        path_name += 1
    number += 1
