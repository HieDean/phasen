import zipfile
import os

path = '../datasets'
namelist = [
	path+'/clean_trainset_wav',
	path+'/noisy_trainset_wav',
]

# unzip
for item in namelist:

	z_file = zipfile.ZipFile(item+".zip", "r")

	if os.path.isdir(item+"/"):
		pass
	else:
		os.mkdir(item+"/")
	print("unzipping"+item+".zip")
	for file in z_file.namelist():
		z_file.extract(file, item+'/')

	z_file.close()
	print("unzip finished")

namelist = [
	path + '/clean_testset_wav',
	path + '/noisy_testset_wav',
]

# unzip
for item in namelist:

	z_file = zipfile.ZipFile(item+".zip", "r")

	if os.path.isdir(item+"/"):
		pass
	else:
		os.mkdir(item+"/")
	print("unzipping"+item+".zip")
	for file in z_file.namelist():
		z_file.extract(file, path+'/')

	z_file.close()
	print("unzip finished")