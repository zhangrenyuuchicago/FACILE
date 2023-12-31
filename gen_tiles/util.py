import openslide
import PIL
import sys
import numpy as np
import os
import ntpath
import scipy.ndimage as snim
import random

# grid size in 20X magnification slide
grid_size = 224
re_size = 224
# white cutoff
hi_cutoff = 0.85
lo_cutoff = 0.10
# keep only those patches within 90% of grayscale intensity distribution
inter_intensity = 1.0
# tile_num to keep
sample_num = 1000

def gen_tile(slidefile, output_folder):
    
    boxname = os.path.basename(os.path.dirname(slidefile))
    boxname = boxname.replace(' ', '')
    if output_folder.endswith('/'):
        output_folder = output_folder[:-1]
    output_folder = output_folder + '/' + boxname + '/'
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    try:
        slide = openslide.open_slide(slidefile)
    except:
        print(f'can not open file: {slidefile}')
        return

    #level = slide.level_count - 1
    width, height = slide.level_dimensions[0]

    file_name = ntpath.basename(slidefile)
    base_name = os.path.splitext(file_name)[0]

    if 'openslide.objective-power' not in slide.properties:
        print("do not have openslide.objective-power")
        return 

    # chop each ROI into 300x300 squares
    if slide.properties['openslide.objective-power'] == '40':
        size = (2*grid_size, 2*grid_size)
    elif slide.properties['openslide.objective-power'] == '20':
        size = (grid_size,grid_size)
    else:
        print("slide openslide.objective-power: %s" % slide.properties['openslide.objective-power'])
        print("magnification not specified")
        return 

    mag = slide.properties['openslide.objective-power']

    #print('begin to clip')
    
    pos_lt = []

    x_pos, y_pos = 0, 0
    while x_pos + size[0] < width:
        while y_pos + size[1] < height:
            try:
                crop_image = slide.read_region((x_pos, y_pos), 0, size) 
            except:
                y_pos += size[1]
                print('read region bug caught')
                continue

            outfile_name = output_folder + "/" +  base_name + "_mag" + mag + "_xpos" + str(x_pos) + "_ypos" + str(y_pos) + ".jpg"
            crop_image = crop_image.resize((re_size, re_size), resample=PIL.Image.ANTIALIAS)
            crop_image = crop_image.convert('RGB')
            gray_image = crop_image.convert('L')
            gray_image_np = np.array(gray_image, dtype=np.float32)
            intensity = np.mean(gray_image_np)/255.0
            if intensity >= lo_cutoff and intensity <= hi_cutoff:
                #crop_image.save( outfile_name )
                pos_lt.append((x_pos, y_pos))

            y_pos += size[1]

        y_pos = 0
        x_pos += size[0]
    
    if len(pos_lt) > sample_num:
        sub_pos_lt = random.sample(pos_lt, sample_num)
        pos_lt = sub_pos_lt

    for pos_pair in pos_lt:
        x_pos, y_pos = pos_pair
        try:
            crop_image = slide.read_region((x_pos, y_pos), 0, size) 
        except:
            print('read region bug caught')
            continue
        
        outfile_name = output_folder + "/" +  base_name + "_mag" + mag + "_xpos" + str(x_pos) + "_ypos" + str(y_pos) + ".jpg"
        crop_image = crop_image.resize((re_size, re_size), resample=PIL.Image.ANTIALIAS)
        crop_image = crop_image.convert('RGB')
        crop_image.save( outfile_name )


