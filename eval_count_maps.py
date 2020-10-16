import numpy as np
import os, sys, csv, cv2, time
import matplotlib.pyplot as plt
from sklearn.externals import joblib 
from skimage.util import view_as_windows
from keras.models import load_model

def _show_countmaps(pred,true,diff):
    f, ax = plt.subplots(1,3)
    ax1, ax2, ax3 = ax.flatten()
    ax1.imshow(pred)
    ax1.set_title('pred')
    ax2.imshow(true)
    ax2.set_title('true')
    ax3.imshow(diff)
    ax3.set_title('diff')
    plt.show()
    
        
if __name__ == "__main__":
    start_time = time.time()  
    
    which_dataset = 'mbm' #'adipocyte' #'vgg'
    file_names = sorted(os.listdir('input/{}/clean_testset'.format(which_dataset)))
    
    models = ['random46.h5','random99.h5','random27.h5']
    models = ['random46.h5']
    models_mae = []
    
    for model_id in models:
        mean_abs_error = []
        model = load_model('checkpoints/{}/{}'.format(which_dataset,model_id))
        patch_size = int(model.inputs[0].shape[1]) 
        
        for i,filename in enumerate(file_names):
            print()
            print('Model {}, Processing image {} (number {})'.format(model_id,filename,i))
            print()
            image_time = time.time()
                
            clean_filename = 'input/{}/clean_testset/{}'.format(which_dataset,filename)
            blackdotted_filename = 'input/{}/blackdotted_testset/{}'.format(which_dataset,filename)
                       
            clean_image = cv2.imread(clean_filename)  
            clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
            clean_image = clean_image/255
            blackdotted_image = cv2.imread(blackdotted_filename,0)
            
            #print(clean_image.shape)
            #print(blackdotted_image.shape)
            #_show_patch(clean_image,blackdotted_image)
            #print(np.max(blackdotted_image))  #check if threshold it's the same
            #continue
            
            #blackdotted_image = cv2.threshold(blackdotted_image, 75, 255, cv2.THRESH_BINARY)[1]   
            blackdotted_image = blackdotted_image/255
            
            total_counts = np.sum(blackdotted_image)
            
            #print('PRE PADDING CLEAN {}'.format(clean_image.shape))   
            #print('PRE PADDING BLACKDOTTED {}'.format(blackdotted_image.shape)) 
            #print()        
                         
            clean_image = np.pad(clean_image, ((patch_size-1,patch_size-1),(patch_size-1,patch_size-1),(0,0)), 'constant', constant_values=(0,0))
            blackdotted_image = np.pad(blackdotted_image, ((patch_size-1,patch_size-1),(patch_size-1,patch_size-1)), 'constant', constant_values=(0,0))
                   
            #print('POST PADDING CLEAN {}'.format(clean_image.shape)) 
            #print('POST PADDING BLACKDOTTED {}'.format(blackdotted_image.shape))
            #print() 
            
            clean_patches = view_as_windows(clean_image,(patch_size,patch_size,3))
            clean_patches = clean_patches.reshape((clean_patches.shape[0],clean_patches.shape[1],patch_size,patch_size,3))
            blackdotted_patches = view_as_windows(blackdotted_image,(patch_size,patch_size))
               
            #print('NUMBER OF CLEAN PATCHES {}'.format(clean_patches.shape))
            #print('NUMBER OF BLACKDOTTED PATCHES {}'.format(blackdotted_patches.shape))
            #print()

            clean_patches = clean_patches.reshape(clean_patches.shape[0]*clean_patches.shape[1],clean_patches.shape[2],clean_patches.shape[3],clean_patches.shape[4])
            
            #print('FLATTEN OF CLEAN PATCHES {}'.format(clean_patches.shape))
            #print()
            
            pred_countmap = model.predict(clean_patches).reshape((blackdotted_patches.shape[0],blackdotted_patches.shape[1]))        
            true_countmap = np.zeros((blackdotted_patches.shape[0],blackdotted_patches.shape[1]))
            
            for r in range(blackdotted_patches.shape[0]):
                for c in range(blackdotted_patches.shape[1]):
                    true_countmap[r,c] = np.sum(blackdotted_patches[r,c,:,:])
            
            pred_countmap = pred_countmap/(patch_size*patch_size)
            true_countmap = true_countmap/(patch_size*patch_size)
            
            #print('PRED COUNTMAP SHAPE {}'.format(pred_countmap.shape))
            #print('TRUE COUNTMAP SHAPE {}'.format(true_countmap.shape))
            #print()
            
            true_count = np.sum(true_countmap)
            pred_count = np.sum(pred_countmap)
            mean_abs_error.append(abs(true_count-pred_count))
                           
            print('TOTAL COUNTS IN IMAGE {}'.format(total_counts))
            print('TRUE_COUNTMAP: TOTAL {} | MAX {} | MIN {}'.format(true_count,true_countmap.max(),true_countmap.min()))
            print('PRED_COUNTMAP: TOTAL {} | MAX {} | MIN {}'.format(pred_count,pred_countmap.max(),pred_countmap.min()))
            print()
            
            #_show_countmaps(pred_countmap,true_countmap,np.abs(pred_countmap - true_countmap))
            
            elapsed = (time.time() - image_time)
            print('Image Processed in {} seconds'.format(elapsed))
            print('-'*30)
            print()
        
        models_mae.append(np.mean(mean_abs_error))    
        print('MEAN ABSOLUTE ERROR OF {} ON TEST SET: {}'.format(model_id,np.mean(mean_abs_error)))
        print()      
     
    print('MAE LIST {}'.format(models_mae))
    print('MAE MEAN {}'.format(np.mean(models_mae)))
    print('MAE STD {}'.format(np.std(models_mae)))
    print()   
    
    elapsed = (time.time() - start_time)/60
    print()
    print('Total Time: {}'.format(elapsed)) 
    
    
    
    
