import os

image_dir = 'C:/Users/wps46139/Documents/Data/PCA/lfw_funneled'
file = open("lfw_names.txt", "w+")

nimg = 0
for subdir in os.listdir(image_dir):
    if subdir.endswith('.txt'):
        continue
#    print(subdir, nimg)
    file.write('%s %d\n' % (subdir, nimg))
    fulldir = image_dir + '/' + subdir
    for filename in os.listdir(fulldir):
        if not filename.endswith('.jpg'):
            continue
        fullname = fulldir + '/' + filename
        nimg += 1

file.close()
print('done')