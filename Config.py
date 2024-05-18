


Initial_params = dict(DataBase_directory = "E:\Bazy_Danych\MNIST_Data",
                      Kaggle_set = True,
                      Load_from_CSV = True,
                      Stratification_test = False,
                      grayscale = True,
                      img_H = 28,
                      img_W = 28,
                      DataType = "float32"
                      )

#If train_dataset_multiplier set to one then there is no augmentation
Augment_params = dict(reduced_set_size = None,
                      dataset_multiplier = 1,
                      flipRotate = False,
                      randBright = False,
                      gaussian_noise = False,
                      denoise = False,
                      contour = False
                      )


Model_parameters = dict(generator_architecture = "Test_generator_28",
                        discriminator_architecture = "Test_discriminator_28",
                        device = "GPU:0",
                        train = True,
                        epochs = 100,
                        latent_dim = 100,
                        batch_size = 256,
                        sample_interval = 1,
                        evaluate = True,
                        show_architecture = False
                       )


