class args():

	# training args
	epochs = 4 #"number of training epochs, default is 2"
	save_per_epoch = 1
	batch_size = 4 #"batch size for training, default is 4"
	dataset = "./train.txt"
	HEIGHT = 256
	WIDTH = 256

	save_model_dir = "./models/" #"path to folder where trained model with checkpoints will be saved."

	image_size = 256 #"size of training images, default is 256 X 256"
	ssim_weight = [1,10,100,1000,10000]

	lr = 1e-4 #"learning rate, default is 0.001"	
	resume = None # if you have, please put the path of the model like "./models/densefuse_gray.model"

	strategy_type = "addition"
	test_save_dir = "./results/"
	test_img = "./img.txt"
	test_ir = "./ir.txt"