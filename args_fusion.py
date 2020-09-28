class args():

	# training args
	epochs = 10 #"number of training epochs, default is 2"
	save_per_epoch = 1
	batch_size = 48 #"batch size for training, default is 4"
	dataset = "/home/oem/shk/dataset/coco_train.txt"
	HEIGHT = 256
	WIDTH = 256
	lr = 1e-4 #"learning rate, default is 0.001"	
	resume = None # if you have, please put the path of the model like "./models/densefuse_gray.model"
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
	strategy_type = "addition"
	test_save_dir = "./results/"
	test_img = "./img.txt"
	test_ir = "./ir.txt"