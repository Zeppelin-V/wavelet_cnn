import train
from config import config

#Dispatch function
def run():


    transform, computing_device, extras, seed = train.init(config['seed'])

    if config['test'] is False:

        train.train(config['model_name'], seed, computing_device,
                    config['num_epochs'], config['k'], config['learning_rate'], config['batch_size'],
                    config['num_mb'], config['wvlt_transform'], config['p_test'], transform, extras, config['outname'])

    else:

        #TODO: Implement this. Test is not functional
        train.test()


if __name__== "__main__":

    run()

