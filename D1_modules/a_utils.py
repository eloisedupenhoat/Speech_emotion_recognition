##### OBJ: all useful functions (e.g. to drwa graphs) #####

def emotion(name):
    filename = name.split('/')[-1]
    filename_no_ext = filename.replace('.jpg', '')
    parts = filename_no_ext.split('-')
    emotion_code = int(parts[2])
    return emotion_code

if __name__ == '__main__':
    pass
