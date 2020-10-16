import numpy as np
import os, sys, csv, cv2, time, random
import matplotlib.pyplot as plt
from sklearn.externals import joblib 
from sklearn.model_selection import train_test_split
from skimage.util import view_as_windows

def _show_patch(clean_patch,blackdotted_patch):
    f, ax = plt.subplots(1,2)
    ax1, ax2 = ax.flatten()
    ax1.imshow(clean_patch)
    ax1.set_title('clean')
    ax2.imshow(blackdotted_patch,cmap='gray')
    ax2.set_title('blackdotted')
    plt.show()

    
if __name__ == "__main__":
    start_time = time.time()  
    
    random_state = 46 #46 #99 #27
    which_dataset = 'mbm' #'adipocyte' #'vgg'
    file_names = sorted(os.listdir('input/{}/clean'.format(which_dataset)))
 
    #for name in file_names:
    #    os.rename('input/clean/'+name, 'input/clean/'+ name.replace('cell',''))
    #sys.exit()
    
    if False:
        for filename in file_names:
            clean_filename = 'input/{}/clean/{}'.format(which_dataset,filename)
            blackdotted_filename = 'input/{}/blackdotted/{}'.format(which_dataset,filename)
                       
            clean_image = cv2.imread(clean_filename)  
            clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
            #clean_image = clean_image/255
            blackdotted_image = cv2.imread(blackdotted_filename,0)
            #blackdotted_image = cv2.threshold(blackdotted_image, 75, 255, cv2.THRESH_BINARY)[1]
            #blackdotted_image = blackdotted_image/255  
               
            #print(clean_image.shape)
            #print(blackdotted_image.shape)          
            #_show_patch(clean_image,blackdotted_image)
            #print(np.max(blackdotted_image))  #check if threshold it's the same   
            #print()                
        sys.exit()    
        
    if False:
        counts = []
        for filename in file_names: 
            blackdotted_filename = 'input/{}/blackdotted/{}'.format(which_dataset,filename)
            blackdotted_image = cv2.imread(blackdotted_filename,0)
            #blackdotted_image = cv2.threshold(blackdotted_image, 75, 255, cv2.THRESH_BINARY)[1]
            blackdotted_image = blackdotted_image/255  
            counts.append(np.sum(blackdotted_image))
        print('MEAN TOTAL COUNTS {}'.format(np.mean(counts)))
        print('STD TOTAL COUNTS {}'.format(np.std(counts)))
        sys.exit()
    
    train_file_names, validation_file_names = train_test_split(file_names, test_size=0.5, random_state=random_state)
    
    #print(len(file_names))    
    #print(len(train_file_names))
    #print(len(validation_file_names))
    #sys.exit()
    
    for which_split in ['train','validation']:
        if which_split=='train':
            file_names = train_file_names
        else:
            file_names = validation_file_names
                       
        patch_size = 32
        patch_counts_dict = {}
        for filename in file_names:
            image_time = time.time()
                
            clean_filename = 'input/{}/clean/{}'.format(which_dataset,filename)
            blackdotted_filename = 'input/{}/blackdotted/{}'.format(which_dataset,filename)

            clean_image = cv2.imread(clean_filename)  
            clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
            blackdotted_image = cv2.imread(blackdotted_filename,0)
            #blackdotted_image = cv2.threshold(blackdotted_image, 75, 255, cv2.THRESH_BINARY)[1]    
             
            #print(clean_image.shape)
            #print(blackdotted_image.shape)
            #print(clean_image.dtype)
            #print(blackdotted_image.dtype)
            #print()
            
            if which_dataset=='vgg':
                step = 7
            elif which_dataset=='adipocyte':
                step = 5
            elif which_dataset=='mbm':
                step = 10
                    
            clean_patches = view_as_windows(clean_image,(patch_size,patch_size,3),step=step)
            blackdotted_patches = view_as_windows(blackdotted_image,(patch_size,patch_size),step=step)
            print(clean_patches.shape)
            print(blackdotted_patches.shape)
                
            for r in range(clean_patches.shape[0]):
                for c in range(clean_patches.shape[1]):
                    n_dots_in_patch = np.sum(blackdotted_patches[r,c,:,:]/255)            
                    patchname = filename.partition('.')[0]+'_'+str(r)+'_'+str(c)+'.png'
                    cv2.imwrite('input/{}/patches32_{}/clean/{}'.format(which_dataset,which_split,patchname),clean_patches[r,c,0,:,:,:])
                    cv2.imwrite('input/{}/patches32_{}/blackdotted/{}'.format(which_dataset,which_split,patchname),blackdotted_patches[r,c,:,:]) 
                    patch_counts_dict[patchname] = n_dots_in_patch
                                      
            elapsed = (time.time() - image_time)
            print('Image {}_{} Processed in {} seconds'.format(which_split,filename,elapsed))
            print()        
            
        joblib.dump(patch_counts_dict,'input/{}/patches32_{}_counts.pkl'.format(which_dataset,which_split))   
        
    elapsed = (time.time() - start_time)/60
    print()
    print('Total Time: {}'.format(elapsed)) 
    
    
    
    
