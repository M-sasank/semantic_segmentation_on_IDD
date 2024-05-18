# Semantic Segmentation On IDD
This repository consists the codes of my Final Year Project at IIITDM Kancheepuram on Semantic Segmentation on IDD

There are two parts in this repository.
* Efficient-UNet implementation (jupyter notebook in root)
* Intern Image implementation (inside segmentation folder)

## Training EffUNet on IDD
1. First create a conda environment:
```conda create -n effunet python=3.8```
```conda activate effunet```

2. Then install the requirements from requirements.txt with the following command:

``` pip install -r requirements.txt ```

3. Download IDD dataset part 1 from https://idd.insaan.iiit.ac.in/ and extract the contents
4. Modify the paths inside the code corresponding to the dataset location and saved models location
5. Run the jupyter notebook from first cell in order.
