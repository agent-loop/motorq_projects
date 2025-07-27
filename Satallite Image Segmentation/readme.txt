1) Step 1 : Get raw data and place it inside data/raw_training_data/raw_data
2) Step 2 : Run step1_pre_process.py to resize, format covert of data into required data.
3) Step 3 : Run labelme "labelme" in bash. , open directory data/raw_training_data/raw_data, annotate every
            image and save its .json file in data/raw_training_data/json_files.
4) Step 4 : Run step2_segmentationmasks.py to convert .json files to mask files and save it in data/raw_training_data/masks
5) Step 5 : Run step3_argumentation.py to perform argumentation on images and masks and save it in same folder.
6) Step 6 : Run step4_splitting.py to divide data into test train val in 1:16:3 ratio.
7) Step 7 : Train model using step5_train.py script, output will be saved in trained_model_outputs.
 