PATH_LOCAL_DATA = 'data/'
PATH_LOCAL_EXPERIMENTS = 'experiment/'

PATH_S3_DATA = 's3://mobility-research/project-dpgan-1/data'
PATH_S3_EXPERIMENTS = 's3://mobility-research/project-dpgan-1/experiment'


# GAN Configuration (for grid search)
dir_user_input = 'users'
dir_user_output = 'users_gan'
user_files = [
    'data_user_002.csv'
#    'data_user_046.csv'
#    'data_user_077.csv'
#    'data_user_099.csv'
#    'data_user_120.csv'
]
input_dim = 3
random_dim = 100
discriminatorDims = [
            [64, 32, 16, 1],
            [128, 64, 32, 16, 1],
#            [256, 128, 64, 32, 16, 1],
#            [512, 256, 128, 64, 32, 16, 1]
        ]
generatorDims = [
            [512, input_dim],
            [128, 64, input_dim],
#            [256, 128, 64, input_dim],
#            [512, 256, 128, 64, input_dim]
        ]        
optimizers = ['Adam']        
batch_sizes = [64, 128]        
epochs = [20] #, 10, 50, 100, 500]
upload_to_s3 = False