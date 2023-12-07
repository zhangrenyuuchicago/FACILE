import os
import glob
import shutil
import subprocess
import gen_feature

for path in glob.glob('manifest/*txt'):
    basename = os.path.basename(path)
    basename = basename[:-4]
    array = basename.split('_')
    organ_name = '_'.join(array[2:])
    
    print(f'organ: {organ_name}')

    if os.path.exists(f'download_slide/{organ_name}/'):
        shutil.rmtree(f'download_slide/{organ_name}/')

    os.mkdir(f'download_slide/{organ_name}/')    
    
    if os.path.exists(f'tiles/{organ_name}'):
        shutil.rmtree(f'tiles/{organ_name}')

    os.mkdir(f'tiles/{organ_name}')
    
    subprocess.run(['./gdc-client', 'download', '-m', path, '-d', f'download_slide/{organ_name}/'])

    gen_feature.gen_tiles(f'download_slide/{organ_name}/', f'tiles/{organ_name}')  

    shutil.rmtree(f'download_slide/{organ_name}/')

