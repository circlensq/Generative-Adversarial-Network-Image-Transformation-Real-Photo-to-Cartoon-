# Image-to-image domain translation using GAN

## Getting Started (Training)
Steps:
1. Download all of the files and folders in this repo and prepare the dataset. In my project, in this project we used [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and Bitmoji dataset run `python create_emojis.py` and set the number of bitmoji images on the `num_emojis` variable.

2. Put the training CelebA dataset inside `dataset/CelebA/trainA/` folder, and test CelebA dataset inside `dataset/CelebA/test`.

3. Put all the Bitmoji dataset inside `dataset/Bitmoji` folder.

4. Set up the config file inside `configs/cifar.json`. Generally, You can determine the number of epochs, n_save_steps, and batch_size. I use `batch_size=32` for faster converged.

5. Run program using command 
```
python train.py --log log_photo2emoji --project_name photo2emoji  
```

## Testing
Steps:
1. Change the `saved_model` key in `config.json` to be `./log_photo2emoji/model_500.pt` or whenever number of iteration model you use.

2. run program using command
```
python testAtoB.py --project_name photo2emoji --log log_photo2emoji
```
