import numpy as np
import os, sys, csv, cv2, time, random, keras, h5py
import skimage.feature
import matplotlib.pyplot as plt
from sklearn.externals import joblib 
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, concatenate, LeakyReLU
from keras.optimizers import Adam


def _show_patch(clean_patch,blackdotted_patch):
    f, ax = plt.subplots(1,2)
    ax1, ax2 = ax.flatten()
    ax1.imshow(clean_patch)
    ax1.set_title('clean')
    ax2.imshow(blackdotted_patch,cmap='gray')
    ax2.set_title('blackdotted')
    plt.show()

def sample_patch(file_names,labels_dict,batch_size,patches_mean,patches_std,initial_path):
    rnd_index = random.randrange(len(file_names))
    filename = file_names[rnd_index]
    path = initial_path+'/'+filename
    patch = cv2.imread(path)
    #patch = (patch-patches_mean)/patches_std
    label = labels_dict[filename]
    #print('Selected patch {} with shape {}'.format(filename,patch.shape))
    return (patch/255).astype('float32'), np.array(label).reshape(1,1,1)
    
def get_batch(file_names,labels_dict,batch_size,patches_mean,patches_std,initial_path):
    x = []
    y = []
    patch, label = sample_patch(file_names,labels_dict,batch_size,patches_mean,patches_std,initial_path)    
    x.append(patch)
    y.append(label)
    return np.array(x),np.array(y)
    
def my_generator(file_names,labels_dict,batch_size,patches_mean,patches_std,initial_path):
    print('\nA generator has been allocated!')
    while True:
        #start_time = time.time()
        #print('\nExtracting Batch...')
        x,y = get_batch(file_names,labels_dict,batch_size,patches_mean,patches_std,initial_path)
        #elapsed = time.time() - start_time
        #print('\nCounts in batch: {} +/- {}'.format(y.mean(),y.std()))
        #print('\nDone in {} seconds...'.format(elapsed))
        yield (x,y)
        
def build_model_patch32(): 
    input_layer = Input(shape=(32,32,3))
    #net = BatchNormalization()(input_layer)
    
    net = Conv2D(64, (3,3), padding='valid', activation=None)(input_layer)
    net = LeakyReLU()(net)
    #net = BatchNormalization()(net)
    
    stack_1 = Conv2D(16, (3,3), padding='same', activation=None)(net)
    #stack_1 = LeakyReLU()(stack_1)
    #stack_1 = BatchNormalization()(stack_1)
    stack_2 = Conv2D(16, (1,1), padding='same', activation=None)(net)
    #stack_2 = LeakyReLU()(stack_2)
    #stack_2 = BatchNormalization()(stack_2)
    net = concatenate([stack_1,stack_2])
    net = LeakyReLU()(net)
    #net = BatchNormalization()(net)

    stack_3 = Conv2D(32, (3,3), padding='same', activation=None)(net)
    #stack_3 = LeakyReLU()(stack_3)
    #stack_3 = BatchNormalization()(stack_3)
    stack_4 = Conv2D(16, (1,1), padding='same', activation=None)(net)
    #stack_4 = LeakyReLU()(stack_4)
    #stack_4 = BatchNormalization()(stack_4)
    net = concatenate([stack_3,stack_4])
    net = LeakyReLU()(net)
    #net = BatchNormalization()(net)  

    net = Conv2D(16, (14,14), padding='valid', activation=None)(net)
    net = LeakyReLU()(net)
    net = BatchNormalization()(net)

    stack_5 = Conv2D(48, (3,3), padding='same', activation=None)(net)
    #stack_5 = LeakyReLU()(stack_5)
    #stack_5 = BatchNormalization()(stack_5)
    stack_6 = Conv2D(112, (1,1), padding='same', activation=None)(net)
    #stack_6 = LeakyReLU()(stack_6)
    #stack_6 = BatchNormalization()(stack_6)
    net = concatenate([stack_5,stack_6])
    net = LeakyReLU()(net)
    #net = BatchNormalization()(net)  

    stack_7 = Conv2D(40, (3,3), padding='same', activation=None)(net)
    #stack_7 = LeakyReLU()(stack_7)
    #stack_7 = BatchNormalization()(stack_7)
    stack_8 = Conv2D(40, (1,1), padding='same', activation=None)(net)
    #stack_8 = LeakyReLU()(stack_8)
    #stack_8 = BatchNormalization()(stack_8)
    net = concatenate([stack_7,stack_8])
    net = LeakyReLU()(net)
    #net = BatchNormalization()(net) 

    stack_9 = Conv2D(96, (3,3), padding='same', activation=None)(net)
    #stack_9 = LeakyReLU()(stack_9)
    #stack_9 = BatchNormalization()(stack_9)
    stack_10 = Conv2D(32, (1,1), padding='same', activation=None)(net)
    #stack_10 = LeakyReLU()(stack_10)
    #stack_10 = BatchNormalization()(stack_10)
    net = concatenate([stack_9,stack_10])
    net = LeakyReLU()(net)
    #net = BatchNormalization()(net) 

    net = Conv2D(64, (17,17), padding='valid', activation=None)(net)
    net = LeakyReLU()(net)
    #net = BatchNormalization()(net)    #NAN 

    net = Conv2D(1, (1,1), padding='valid', activation=None)(net)
    net = LeakyReLU()(net)
    #net = BatchNormalization()(net)
    
    net = Conv2D(1, (1,1), padding='valid', activation=None)(net)
    #net = LeakyReLU()(net)
    #net = BatchNormalization()(net)
    
    model = Model(inputs=input_layer, outputs=net)
    
    #model.summary()
    
    #print('*'*30)
    #for l in model.layers:
    #    print('-'*30)
    #    print(l.input_shape)
    #    print(l.output_shape)
    #    print('-'*30)
    #print('*'*30)
    
    optimizer = Adam(lr=0.001)
    model.compile(loss='mean_absolute_error',
                  optimizer=optimizer,
                  metrics=[])
    return model    

    
if __name__ == "__main__":
    start_time = time.time()  
    
    which_dataset = 'mbm' #'adipocyte' #'vgg'
    train_file_names = os.listdir('input/{}/patches32_train/clean'.format(which_dataset))
    validation_file_names = os.listdir('input/{}/patches32_validation/clean'.format(which_dataset))
    
    train_labels_dict = joblib.load('input/{}/patches32_train_counts.pkl'.format(which_dataset))
    validation_labels_dict = joblib.load('input/{}/patches32_validation_counts.pkl'.format(which_dataset))
    
    # get mean and std for centering
    clean_patches = np.array([cv2.imread('input/{}/patches32_train/clean/{}'.format(which_dataset,filename)) for filename in train_file_names])
    patches_mean = np.mean(clean_patches,axis=0)
    patches_std = np.std(clean_patches,axis=0)
    
    print()
    print('Mean counts in train_set {}'.format(np.array(list(train_labels_dict.values())).mean()))
    print('Mean counts in validation_set {}'.format(np.array(list(validation_labels_dict.values())).mean()))
    print('Std counts in train_set {}'.format(np.array(list(train_labels_dict.values())).std()))
    print('Std counts in validation_set {}'.format(np.array(list(validation_labels_dict.values())).std()))
    print()
    
    batch_size = 32
    modelid = time.strftime('%Y%m%d%H%M%S')

    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=200),
        keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/{}/model_checkpoint_best_{}.h5'.format(which_dataset,modelid),
            monitor='val_loss',
            save_best_only=True),
        keras.callbacks.TensorBoard(
            log_dir='./logs/{}/{}'.format(which_dataset,modelid),
            histogram_freq=0, write_graph=False, write_images=False)
    ]

    model = build_model_patch32()
    
    history=model.fit_generator(
                        generator=my_generator(train_file_names,train_labels_dict,batch_size,patches_mean,patches_std,'input/{}/patches32_train/clean'.format(which_dataset)),
                        use_multiprocessing=False,
                        workers=1,
                        steps_per_epoch=500, 
                        epochs=10000, 
                        verbose=1,
                        validation_data=my_generator(validation_file_names,validation_labels_dict,batch_size,patches_mean,patches_std,'input/{}/patches32_validation/clean'.format(which_dataset)),
                        validation_steps=500,
                        callbacks=callbacks_list)
        
    elapsed = (time.time() - start_time)/60
    print()
    print('Total Time: {}'.format(elapsed)) 
    
    
    
    
