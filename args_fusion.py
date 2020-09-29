class args():

	# training args
	epochs = 10 #"number of training epochs, default is 2"
	save_per_epoch = 1
	batch_size = 48 #"batch size for training, default is 4"
	dataset = "./train.txt"
	HEIGHT = 256
	WIDTH = 256
	CHANNELS = 3 # grayscale: 1, RGB: 3
	lr = 1e-4 #"learning rate, default is 0.001"	
	# resume = "models/rgb.pt" # if you have, please put the path of the model like "./models/densefuse_gray.model"
	resume = None
	ssim_weight = [1,10,100,1000,10000]
	save_model_dir = "./models/" #"path to folder where trained model with checkpoints will be saved."

	# For GPU training
	world_size = -1
	rank = -1
	dist_backend = 'nccl'
	gpu = 0,1,2,3
	multiprocessing_distributed = True
	distributed = None

	# For testing
	strategy_type = "attention"
	test_save_dir = "./"
	test_img = "./test_rgb.txt"
	test_ir = "./test_ir.txt"