from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='speech_emotion_recognition',
      version="1.0",
      description="Emotion_recognition_model",
      author="Le Wagon_batch1976",
      #url="https://github.com/eloisedupenhoat/Speech_emotion_recognition",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      include_package_data=True,
      zip_safe=False)
