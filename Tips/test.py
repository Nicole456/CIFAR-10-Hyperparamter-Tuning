import os

for root, dirs, files in os.walk('J:/temp/all'):
    for dir in dirs:
        print(dir)
