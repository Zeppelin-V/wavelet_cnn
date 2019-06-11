import os
import sys
import pandas as pd



def make_csv(img_dir, csv_out):


    working_dir = os.getcwd()

    dir_name = os.path.join(working_dir, img_dir)

    #List to hold amount of images in each class
    class_counts = []

    #Lists to hold filename and label of each image
    image_names = []
    image_labels = []


    #Get each class and the number of images in that class
    for class_dir in os.listdir(dir_name):

        class_path = os.path.join(dir_name, class_dir)

        class_images = os.listdir(class_path)

        class_counts.append((class_dir, len(class_images)))

    #Sort by size of each class and then get largest ten classes
    sorted_class_counts = sorted(class_counts, key=lambda x:x[1], reverse=True)

    largest_ten_classes = [sorted_class_counts[i][0] for i in range(0,10)]

    #Add each class only to the list if it is in the top ten classes
    for class_dir in os.listdir(dir_name):

        if class_dir in largest_ten_classes:

            for image in os.listdir(os.path.join(dir_name, class_dir)):

                image_names.append(image)
                image_labels.append(class_dir)


    #Output as csv
    image_label_df = pd.DataFrame(list(zip(image_names, image_labels)), columns=['name', 'labels'])

    outpath = os.path.join(working_dir, csv_out)

    image_label_df.to_csv(outpath)



if __name__ == '__main__':


    img_dir = sys.argv[1]
    csv_out = sys.argv[2]

    make_csv(img_dir, csv_out)

