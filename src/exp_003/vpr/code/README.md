# Train and test

To run the experiments, you have to:
1. have alderley_paired, Tokyo_24_7 and gta folders with datasets. GTA V must be resized with the script in *dataset/image_resizer.py*. Check if the alderley images are of shape 462x260, Tokyo_24_7 of shape 224x224 and gta images of shape 398x224
2. To train the nn, run the script *train.py*. Change the parameters in *train.yaml*.
3. To test the nn, run the script *test.py*. Change the parameters in *test.yaml*. This script simply create matrices in the experiment folder.
4. To compute the recall, run the script *recall.py*. Change the parameters in *recall.yaml*. This script simply create txt files with the recall in the experiment folder.