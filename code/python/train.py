import numpy as np
import librosa
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


# 提取特征
def extract_feature(file_path):
    y,sr = librosa.load(file_path,duration=2.5,sr=22050)
    mfccs = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13)
    return np.mean(mfccs.T,axis=0)


audio_features = ["../profile/wav_0/流行a.wav","../profile/wav_0/流行e.wav","../profile/wav_0/流行i.wav","../profile/wav_0/流行o.wav","../profile/wav_0/流行u.wav"]
labels = [0,1,2,3,4]

features = np.array([extract_feature(file) for file in audio_features])
labels = np.array(labels)

recongnition_model = keras.Sequential([
    keras.layers.Dense(128,activation="relu",input_shape=(features.shape[1],)),
    keras.layers.Dense(64,activation="relu"),
    keras.layers.Dense(5,activation="softmax")
])

recongnition_model.compile(optimizer=keras.optimizers.Adam,loss=keras.losses.sparse_categorical_crossentropy,metrics=[
        keras.metrics.BinaryAccuracy(),
        keras.metrics.FalseNegatives(),
    ],
)

x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=0.2,random_state=42)
recongnition_model.fit(x_train,y_train,epochs=50,batch_size=16,validation_data=(x_test, y_test))

def match_audio(file_path):
    feature = extract_feature(file_path)
    feature = np.reshape(feature,(1,-1)) # 调整形状以适应模型输入
    prediction = recongnition_model.predict(feature)
    recognized_label = np.argmax(prediction)
    return recognized_label

def recongnition_match_audio(test_file_path):
    recognized_label = match_audio(test_file_path)
    recognized_audio = audio_features[recognized_label]  # 获取对应的音频文件
    print(f'Recognized label: {recognized_label}, Audio content: {recognized_audio}')
    
    # 获取对应的标准音频
    standard_audio_path = audio_features[recognized_label]  
    print(f'Matching with standard audio: {standard_audio_path}')

    standard_feature = extract_feature(standard_audio_path)
    test_feature = extract_feature(test_file_path)
    similarity = cosine_similarity([test_feature], [standard_feature])
    print(f'Cosine similarity: {similarity[0][0]}')

