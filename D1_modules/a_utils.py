##### OBJ: all useful functions (e.g. to drwa graphs) #####

def emotion(name):
    filename = name.split('/')[-1]
    filename_no_ext = filename.replace('.jpg', '')
    parts = filename_no_ext.split('-')
    emotion_code = int(parts[2])
    return emotion_code

def decodeur_emotion(emotion_code):
    emotion_code = int(emotion_code)
    emotion_map = {
        1: 'neutral',
        2: 'calm',
        3: 'happy',
        4: 'sad',
        5: 'angry',
        6: 'fearful',
        7: 'disgust',
        8: 'surprised'}
    emotion = emotion_map[emotion_code]
    return emotion

if __name__ == '__main__':
    pass
