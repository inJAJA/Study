
import glob

def file_path_save():
    filenames = []
    files = sorted(glob.glob("D:/YOLO_test/yolo_tran_data/*.jpg"))
    # data = sorted(glob.glob("D:/bit_camp/darknet-master/YOLO_test/train_images/meat/*.txt"))

    for i in range(len(files)):
        f = open("D:/YOLO_test/yolo_tran_data/train_list.txt", 'a')
        # k = open(data[i], 'r')
        # line = k.readline()
        # f.write(files[i] + '\n' + line )
        f.write(files[i] + '\n')

    

if __name__ == '__main__':
    file_path_save()


